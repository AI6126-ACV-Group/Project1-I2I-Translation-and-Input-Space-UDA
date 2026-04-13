import torch.nn as nn
import torch
from torchvision import models as tv_models

class ResNet18Model(nn.Module):
    def __init__(self, num_cls=31, **kwargs):
        super(ResNet18Model, self).__init__()
        # 加载 torchvision 的原生 resnet18
        self.model = tv_models.resnet18(pretrained=True)

        # 替换最后的全连接层以匹配 Office-31 的类别数 (31)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_cls)

        # 必须设置这些属性，因为 AddaNet 的 setup_net 会读取它们
        self.image_size = 128
        self.num_channels = 3

    def forward(self, x, with_ft=False):
        # 如果嵌套在 self.model 中
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)  # 这里应该是 512 维
        score = self.model.fc(feature)  # 这里是 31 维
        if with_ft:
            return score, feature  # 同时返回 score 和 feature
        return score

    def load(self, init_path):
        """适配 AddaNet 中 self.src_net.load(init_path) 的调用"""
        state_dict = torch.load(init_path, map_location='cpu')
        self.load_state_dict(state_dict)