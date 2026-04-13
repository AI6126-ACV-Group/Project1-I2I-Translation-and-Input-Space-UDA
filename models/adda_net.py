import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()

class AddaNet(nn.Module):
    "Defines and Adda Network."
    def __init__(self, src_net, tgt_net, num_cls=31, feature_dim=512,image_size=128,num_channels=3, discrim_feat=True):
        super(AddaNet, self).__init__()
        self.discrim_feat=discrim_feat
        self.name = 'AddaNet'
        self.num_cls = num_cls
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()

        self.src_net = src_net
        self.tgt_net = tgt_net

        input_dim = feature_dim
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
        )

        # 属性同步
        self.image_size = getattr(src_net, 'image_size', image_size)
        self.num_channels = getattr(src_net, 'num_channels', num_channels)

        init_weights(self.discriminator)

    def forward(self, x_s, x_t):
        """Pass source and target images through their
        respective networks."""
        score_s, x_s = self.src_net(x_s, with_ft=True)
        score_t, x_t = self.tgt_net(x_t, with_ft=True)

        if self.discrim_feat:
            d_s = self.discriminator(x_s)
            d_t = self.discriminator(x_t)
        else:
            d_s = self.discriminator(score_s)
            d_t = self.discriminator(score_t)
        return score_s, score_t, d_s, d_t

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_tgt_net(self, out_path):
        """
        保存目标域网络权重，并移除 'model.' 前缀以适配标准测试脚本。
        """
        # 1. 获取 tgt_net 的原始 state_dict (带有 model. 前缀)
        state_dict = self.tgt_net.state_dict()
        # 2. 创建一个新的字典，移除 key 中的 "model." 字符串
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                # 去掉 "model." 这 6 个字符
                new_key = k[6:]
                new_state_dict[new_key] = v
            else:
                # 防止有其他不带前缀的参数（如 fc 层）
                new_state_dict[k] = v
        # 3. 保存这个“干净”的权重
        torch.save(new_state_dict, out_path)
        print(f"Clean weights saved to {out_path}, ready for UDA_test.py")

