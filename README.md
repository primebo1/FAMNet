# FAMNet
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2412.09319)
[![AAAI](https://img.shields.io/badge/AAAI'25-Paper-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/32184)

Official code for AAAI 2025 paper: FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation

- [**News!**] 24-12-10: Our work is accepted by AAAI25. [Arxiv Paper](https://arxiv.org/abs/2412.09319) can be found here. 🎉
- [**News!**] 25-12-31: Check out our [latest CD-FSMIS paper](https://ieeexplore.ieee.org/document/11318044) accepted by IEEE T-MI!


## 💡 Overview of FAMNet 
![](./FAMNet.png)


## 📋 Abstract
Existing few-shot medical image segmentation (FSMIS) models fail to address a practical issue in medical imaging: the domain shift caused by different imaging techniques, which limits the applicability to current FSMIS tasks. To overcome this limitation, we focus on the cross-domain few-shot medical image segmentation (CD-FSMIS) task, aiming to develop a generalized model capable of adapting to a broader range of medical image segmentation scenarios with limited labeled data from the novel target domain.
Inspired by the characteristics of frequency domain similarity across different domains, we propose a Frequency-aware Matching Network (FAMNet), which includes two key components: a Frequency-aware Matching (FAM) module and a Multi-Spectral Fusion (MSF) module. The FAM module tackles two problems during the meta-learning phase: 1) intra-domain variance caused by the inherent support-query bias, due to the different appearances of organs and lesions, and 2) inter-domain variance caused by different medical imaging techniques. Additionally, we design an MSF module to integrate the different frequency features decoupled by the FAM module, and further mitigate the impact of inter-domain variance on the model's segmentation performance.
Combining these two modules, our FAMNet surpasses existing FSMIS models and Cross-domain Few-shot Semantic Segmentation models on three cross-domain datasets, achieving state-of-the-art performance in the CD-FSMIS task.


## ⏳ Quick start

### 🛠 Dependencies
Please install the following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```


### 📚 Datasets and Preprocessing
Please download:
1) **Abdominal MRI**: [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)
2) **Abdominal CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **Cardiac LGE and b-SSFP**: [Multi-sequence Cardiac MRI Segmentation dataset](https://zmiclab.github.io/zxh/0/mscmrseg19/index.html)
4) **Prostate UCLH and NCI**: [Cross-institution Male Pelvic Structures](https://zenodo.org/records/7013610)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their GitHub repository.

### 📦 Pretrained Weights
1. *(Optional)* You can download our [pretrained models](https://drive.google.com/drive/folders/1ER-dAD34UPpTP6ZHfQ4tmGZYV1iB-rF0?usp=sharing) for different domains:

   * **Abdominal CT**: [Google Drive](https://drive.google.com/drive/folders/1vzSSeUVgyOL2WBVSxp35X-WQUOZQ4Kcy?usp=sharing)
   * **Abdominal MRI**: [Google Drive](https://drive.google.com/drive/folders/1-8s2RGthA8iHsrVA2RnYpq3AdIx409Rm?usp=sharing)
   * **Cardiac LGE**: [Google Drive](https://drive.google.com/drive/folders/1xLc0wDQMorLHLehPe_dnV_lxHaFkLqz3?usp=sharing)
   * **Cardiac b-SSFP**: [Google Drive](https://drive.google.com/drive/folders/1xev_e6WU0trRXsexWwOxF9eBtDf19paB?usp=sharing)
    * **Prostate UCLH**: [Google Drive](https://drive.google.com/drive/folders/1XmaiZkyIATJdaQ2gqv-SGqTXI_e4A4P0?usp=sharing)
    * **Prostate NCI**: [Google Drive](https://drive.google.com/drive/folders/1VJI5_eBc7KmY4CinChJ0zgPDBncS6f1Z?usp=sharing)

   After downloading, update the path accordingly in the test script.

### 🔥 Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py`
2. Download the pre-trained [ResNet-50 weights](https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth) and put in your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train_<direction>.sh`, for example: `./script/train_ct2mr.sh`


### 🔍  Inference
Run `./script/test_<direction>.sh` 


## 🥰 Acknowledgements
Our code is built upon the works of [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [ADNet](https://github.com/sha168/ADNet) and [QNet](https://github.com/ZJLAB-AMMI/Q-Net), we appreciate the authors for their excellent contributions!


## 📝 Citation
If you use this code for your research or project, please consider citing our paper. Thanks!🥂:
```bibtex
@inproceedings{bo2025famnet,
  title={FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation},
  author={Bo, Yuntian and Zhu, Yazhou and Li, Lunbo and Zhang, Haofeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={1889-1897},
  year={2025},
  DOI={10.1609/aaai.v39i2.32184}
}
```

