import os
import torch
from options.train_options import TrainOptions
from data import create_dataset
from torchvision import transforms

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # 1. 强制设置参数
    opt.dataset_mode = 'unalignedlabel'
    opt.serial_batches = True
    opt.batch_size = 1  # 核心技巧：测试时把 batch_size 设为 1，绝对不会报 stack 错误

    # 2. 创建数据集
    dataset_loader = create_dataset(opt)
    actual_dataset = dataset_loader.dataset

    # 3. 【暴力修正】直接重写实例的 transform 逻辑
    # 这样可以绕过 get_transform 里的所有 bug
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 对应你设置的 128
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    actual_dataset.transform_A = test_transform
    actual_dataset.transform_B = test_transform

    # 4. 打印映射表
    print("\n" + "=" * 30)
    print("Class to Index Mapping:")
    for cls_name, idx in actual_dataset.class_to_idx.items():
        print(f"ID {idx:2d} : {cls_name}")
    print("=" * 30 + "\n")

    # 5. 遍历检查
    print("Checking First 5 Samples:")
    # 直接迭代底层 dataset 绕过 dataloader 的 stack 过程，更安全
    for i in range(min(20, len(actual_dataset))):
        data = actual_dataset[i]  # 直接调用 __getitem__

        a_path = data['A_paths']  # 注意：直接调用 dataset 时，返回的不是 list
        a_label = data['A_label'].item()

        idx_to_class = {v: k for k, v in actual_dataset.class_to_idx.items()}
        detected_name = idx_to_class[a_label]

        print(f"Sample {i}:")
        print(f"  Path:  {a_path}")
        print(f"  Label: {a_label} (Detected as: {detected_name})")

        if detected_name.lower() in os.path.basename(a_path).lower():
            print("  Result: [OK] ✅")
        else:
            print("  Result: [ERROR] ❌ 标签匹配错误！")
        print("-" * 20)


