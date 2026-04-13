from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import sys
import os
import torch.utils.data as data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_resnet_18 import RescaleShortSide


def get_aligned_transform(load_size=128, crop_size=128, is_train=True):
    """
    完全对齐你 ResNet-18 训练时的预处理逻辑
    """
    return transforms.Compose([
        RescaleShortSide(load_size=load_size, crop_size=crop_size, is_train=is_train),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # 对齐 [0.5, 0.5, 0.5] 的归一化
    ])

def load_adda_data_aligned(src_dir, tgt_dir, batch=32, load_size=128, crop_size=128, kwargs={}):
    """
    为 ADDA 训练加载对齐预处理后的源域和目标域数据
    """
    # ADDA 训练时，通常两个域都使用 "train" 模式的增强（包含随机裁剪/翻转）
    # 如果你希望严格保持风格一致，也可以都用 is_train=False
    transform = get_aligned_transform(load_size=load_size, crop_size=crop_size, is_train=True)

    src_dataset = datasets.ImageFolder(root=src_dir, transform=transform)
    tgt_dataset = datasets.ImageFolder(root=tgt_dir, transform=transform)

    print("checking label alignment ...:" )
    print(src_dataset.class_to_idx == tgt_dataset.class_to_idx)
    # 包装成成对的 AddaDataset (利用你代码中已有的 AddaDataset 类)
    #
    dataset = AddaDataset(src_dataset, tgt_dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        **kwargs
    )
    return loader

class AddaDataset(data.Dataset):

    def __init__(self, src_data, tgt_data):
        self.src = src_data
        self.tgt = tgt_data

    def __getitem__(self, index):
        ns = len(self.src)
        nt = len(self.tgt)
        xs, ys = self.src[index % ns]
        xt, yt = self.tgt[index % nt]
        return (xs, ys), (xt, yt)

    def __len__(self):
        return min(len(self.src), len(self.tgt))

