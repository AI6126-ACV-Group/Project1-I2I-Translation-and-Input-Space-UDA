import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
from datetime import datetime

# 这里需要确保 RescaleShortSide 类在当前作用域可用
# 如果你在另一个文件，记得 import 它
from train_resnet_18 import RescaleShortSide

def test_single_model(data_dir, weight_path, out_features, device, batch_size=32, load_size=128, crop_size=128):
    """
    核心测试逻辑：负责单个权重的评估
    """
    # 1. 定义数据预处理
    test_transform = transforms.Compose([
        RescaleShortSide(load_size=load_size, crop_size=crop_size, is_train=False),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # 2. 加载数据集
    test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. 初始化模型
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, out_features)

    # 4. 载入权重并处理前缀
    try:
        state_dict = torch.load(weight_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            # 自动处理 AddaNet 导出的 'model.' 前缀
            new_key = k[6:] if k.startswith("model.") else k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"  [Error] 无法加载权重 {weight_path}: {e}")
        return None

    model.to(device)
    model.eval()

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

    return 100 * correct / total


def run_all_tests():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []

    # 定义所有实验任务
    experiments = [
        {
            "name": "Office-31 (Webcam)",
            "data_dir": 'original_datasets/office_31/webcam',
            "num_cls": 31,
            "weights": [
                "checkpoints/Resnet_REAL_amazon/resnet18_amazon.pth",
                "checkpoints/Resnet_amazon2webcam_cyclegan/resnet18_amazon2webcam_cyclegan.pth",
                "checkpoints/Resnet_amazon2webcam_fgcyclegan/resnet18_amazon2webcam_fgcyclegan.pth",
                "checkpoints/Resnet_amazon2webcam_semcyclegan/resnet18_amazon2webcam_semcyclegan.pth",
                "checkpoints/Resnet_amazon2webcam_cycada/resnet18_adda_optimized.pth",
                "checkpoints/Resnet_amazon2webcam_cycada/resnet18_fg_adda_optimized.pth"
            ]
        },
        {
            "name": "Office-Home (Real World)",
            "data_dir": 'original_datasets/officehome/Real World',
            "num_cls": 65,
            "weights": [
                "checkpoints/Resnet_REAL_art/resnet18_art.pth",
                "checkpoints/Resnet_art2realworld_cyclegan/resnet18_art2realworld_cyclegan.pth",
                "checkpoints/Resnet_art2realworld_fgcyclegan/resnet18_art2realworld_fgcyclegan.pth",
                "checkpoints/Resnet_art2realworld_semcyclegan/resnet18_art2realworld_semcyclegan.pth",
                "checkpoints/Resnet_art2realworld_cycada/resnet18_adda_optimized.pth",
                "checkpoints/Resnet_art2realworld_cycada/resnet18_fg_adda_optimized.pth"
            ]
        },
        {
            "name": "PACS (Sketch)",
            "data_dir": 'original_datasets/PACS/sketch',
            "num_cls": 7,
            "weights": [
                "checkpoints/Resnet_REAL_photo/resnet18_photo.pth",
                "checkpoints/Resnet_photo2sketch_cyclegan/resnet18_photo2sketch_cyclegan.pth",
                "checkpoints/Resnet_photo2sketch_fgcyclegan/resnet18_photo2sketch_fgcyclegan.pth",
                "checkpoints/Resnet_photo2sketch_semcyclegan/resnet18_photo2sketch_semcyclegan.pth",
                "checkpoints/Resnet_photo2sketch_cycada/resnet18_adda_optimized.pth",
                "checkpoints/Resnet_photo2sketch_cycada/resnet18_fg_adda_optimized.pth"
            ]
        }
    ]

    print(f"\n{'=' * 20} 开始全量测试 {'=' * 20}")
    for exp in experiments:
        print(f"\n>> 正在测试任务: {exp['name']}")
        for weight in exp['weights']:
            if not os.path.exists(weight):
                print(f"  [Skip] 文件不存在: {weight}")
                continue

            acc = test_single_model(exp['data_dir'], weight, exp['num_cls'], device)

            if acc is not None:
                print(f"  Acc: {acc:6.2f}% | Path: {weight}")
                # 记录结果数据
                results.append({
                    "Task": exp['name'],
                    "Model_Path": weight,
                    "Accuracy": f"{acc:.2f}%",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })

    # 保存到 CSV 文件
    if results:
        df = pd.DataFrame(results)
        save_path = f"test_results_{datetime.now().strftime('%m%d_%H%M')}.csv"
        df.to_csv(save_path, index=False, encoding='utf_8_sig')
        print(f"\n{'=' * 20} 测试完成 {'=' * 20}")
        print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    run_all_tests()