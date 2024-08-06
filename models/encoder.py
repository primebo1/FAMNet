import torch
import torch.nn as nn
import torchvision

"""
Pretrained Model: ResNet50
Pretrained on Dataset: COCO and ResNet
"""


class Res50Encoder(nn.Module):
    """
    Resnet50 backbone from deeplabv3
    modify the 'downsample' component in layer2 and/or layer3 and/or layer4 as the vanilla Resnet
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights='resnet50'):
        super().__init__()
        # using pretrained model's weights
        if pretrained_weights == 'COCO':
            self.pretrained_weights = torch.load(
                "deeplabv3_resnet50_coco-cd0a2569.pth", map_location='cpu')  # pretrained on COCO
        elif pretrained_weights == 'ImageNet':
            self.pretrained_weights = torch.load(
                "resnet50-19c8e357.pth", map_location='cpu')  # pretrained on ImageNet
        else:
            self.pretrained_weights = pretrained_weights

        _model = torchvision.models.resnet.resnet50(pretrained=False,
                                                    replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self.IN_layer = nn.Sequential(
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def forward(self, x):

        """
        :param x:  (2, 3, 256, 256)
        :return:
        """
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)

        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)
        feature = self.reduce1(x)  # (2, 512, 64, 64)
        feature = self.IN_layer(feature)
        x = self.backbone["layer4"](x)
        # feature map -> avgpool -> fc -> single value
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)
        return (feature, t)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained_weights is not None:
            keys = list(self.pretrained_weights.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(len(keys)):
                if keys[i] in new_keys:
                    new_dic[keys[i]] = self.pretrained_weights[keys[i]]

            self.load_state_dict(new_dic)

