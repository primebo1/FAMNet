#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import ex
from dataloaders.datasets import TrainDataset as TrainDataset
from models.CDFSMIS import FewShotSeg
from utils import *
from thop import profile

def pixel_accuracy(pred, label):
    pred_flatten = pred.flatten()
    label_flatten = label.flatten()
    accuracy = accuracy_score(label_flatten, pred_flatten)
    return accuracy


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model_config = {
        # 'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'PREC': _config['PREC'],
        # 'train_classname': _config['train_classname'],
        # 'test_classname': _config['test_classname'],
        'BACKBONE_NAME': _config['BACKBONE_NAME'],
        'N_CTX': _config['N_CTX'],
        'CTX_INIT': _config['CTX_INIT'],
        'CLASS_TOKEN_POSITION': _config['CLASS_TOKEN_POSITION'],
        'INPUT_SIZE': _config['INPUT_SIZE'],
        'CSC': _config['CSC'],
        'INIT_WEIGHTS': _config['INIT_WEIGHTS'],
        'OPTIM': _config['OPTIM'],
        'PROMPT_INIT': _config['PROMPT_INIT'],
        # 'classnames': _config['classnames'],
    }
    model = FewShotSeg(model_config)
    model = model.cuda()
    model.train()

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)
    # criterion = nn.CrossEntropyLoss()

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
        'train_organ': _config['train_organ'],

    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0, 'thresh_loss': 0}

    loss_values = []
    i_iter = 0
    _log.info(f'Start training...')
    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')
        for _, sample in enumerate(train_loader):

            # Prepare episode data.
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                               for way in sample['support_fg_labels']]

            # prompt = _config['dataset']

            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)


            # Compute outputs and losses.
            query_pred, proto_loss = model(support_images, support_fg_mask, query_images, query_labels, opt=optimizer, train=True)
 
            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)



            loss = query_loss + proto_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()

            loss_values.append(query_loss)

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss

            # Print loss and take snapshots.
            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss},')
                # f' align_loss: {align_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            i_iter += 1

    loss_values = np.array(loss_values)
    # loss_values = loss_values.detach().cpu().numpy()
    np.savetxt('loss_values.txt', loss_values)

    _log.info('End of training.')
    return 1
