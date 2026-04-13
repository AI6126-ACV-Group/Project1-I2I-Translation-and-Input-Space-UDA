import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

# 这里需要确保 RescaleShortSide 类在当前作用域可用
# 如果你在另一个文件，记得 import 它
from train_resnet_18 import RescaleShortSide


def test_model(data_dir, weight_path, out_features=31, batch_size=32, load_size=128, crop_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 定义数据预处理 (保持和训练时的 test_val_transform 一致)
    test_transform = transforms.Compose([
        RescaleShortSide(load_size=load_size, crop_size=crop_size, is_train=False),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # 2. 加载新数据集
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到数据集路径: {data_dir}")

    test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Loaded dataset from {data_dir} with {len(test_dataset)} images.")

    # 3. 初始化模型架构并修改输出层
    model = models.resnet18(weights=None)  # 测试时不需加载预训练权重，因为我们要载入自己的
    model.fc = nn.Linear(model.fc.in_features, out_features)

    # 4. 载入你训练好的权重
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件: {weight_path}")

    # 关键点：map_location 确保在不同设备（CPU/GPU）间转换正常
    state_dict = torch.load(weight_path, map_location=device)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            # 去掉前 6 个字符 ("model.")
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    if 'fc.weight' in new_state_dict:
        print("Found fc weights in pth file, loading...")
    else:
        print("Warning: No fc weights found! Model will use random classification.")

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Successfully loaded weights from {weight_path}")

    # 5. 执行评估
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("-" * 30)
    print(f'Test Accuracy on new dataset: {accuracy:.2f}%')
    print("-" * 30)

    return accuracy


if __name__ == "__main__":
    # 修改以下路径进行测试
    NEW_DATA_DIR = 'original_datasets/office_31/webcam'
    PTH_PATH_list=["checkpoints/Resnet_REAL_amazon/resnet18_amazon.pth",  #48.43%
                   "checkpoints/Resnet_amazon2webcam_cyclegan/resnet18_amazon2webcam_cyclegan.pth", # 53.21%
                   "checkpoints/Resnet_amazon2webcam_fgcyclegan/resnet18_amazon2webcam_fgcyclegan.pth"] # 56.98%

    PTH_PATH_list = ["checkpoints/Resnet_amazon2webcam_cycada/resnet18_adda_optimized.pth"] # 54.84%

    for PTH_PATH in PTH_PATH_list:
        test_model(
            data_dir=NEW_DATA_DIR,
            weight_path=PTH_PATH,
            out_features=31  # 确保类别数与训练时一致
        )
