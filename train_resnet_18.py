import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
import random
import numpy as np
import torch.optim as optim
import logging
from matplotlib import pyplot as plt
import shutil

## 与 cycle Gan 训练时使用相同的rescale 方法
class RescaleShortSide (object):
    def __init__(self, load_size, crop_size, is_train=True):
        self.load_size = load_size
        self.crop_size = crop_size
        self.is_train = is_train

    def __call__(self, img):
        # 1. Rescale Short Side (Zy's Logic)
        w, h = img.size
        if w < h:
            new_w = self.load_size
            new_h = self.load_size * h // w
        else:
            new_h = self.load_size
            new_w = self.load_size * w // h

        img = img.resize((new_w, new_h), Image.BILINEAR)
        # 2. Random Crop / Center Crop (Zy's Logic)

        if self.is_train:
            # 训练模式：随机位置裁切
            x = random.randint(0, np.maximum(0, new_w - self.crop_size))
            y = random.randint(0, np.maximum(0, new_h - self.crop_size))
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

            # 3. Random Flip (Zy's Logic)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # 测试模式：固定中心裁切
            x = (new_w - self.crop_size) // 2
            y = (new_h - self.crop_size) // 2
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

        return img


def prepare_data(data_dir, load_size=128, crop_size=128, batch_size=32):
    # 转换逻辑保持你定义的 RescaleShortSide
    train_transform = transforms.Compose([
        RescaleShortSide(load_size=load_size, crop_size=crop_size, is_train=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    test_val_transform = transforms.Compose([
        RescaleShortSide(load_size=load_size, crop_size=crop_size, is_train=False),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    full_dataset = datasets.ImageFolder(data_dir)
    n_total = len(full_dataset)
    n_test = int(0.2 * n_total)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_test - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # 分别赋予 transform (注意 Subset 对象的特殊处理)
    # 我们通过包装类或直接在 DataLoader 前指定
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = test_val_transform
    test_ds.dataset.transform = test_val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dataset Split: Train={n_train}, Val={n_val}, Test={n_test}")
    return train_loader, val_loader, test_loader

# 配置日志设置
def setup_logger(save_path):
    log_dir = os.path.dirname(save_path)
    # --- 核心修改：如果文件夹存在，先清空再创建 ---
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 彻底删除旧文件夹及其内容
    os.makedirs(log_dir, exist_ok=True)  # 重新创建干净的文件夹

    log_file = os.path.join(log_dir, 'train_log.txt')

    # 彻底重置 root logger，防止重复运行脚本时日志不刷新
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 'w' 模式确保覆盖
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_model(train_loader, val_loader, out_features=31, num_epochs=50, lr=1e-4,
                patience=5, save_path='./checkpoints/resnet18_amazon.pth'):
    logger = setup_logger(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 用于绘图的数据记录
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    logger.info(f"Starting training on {device} | Total Epochs: {num_epochs} | LR: {lr}")
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        current_lr = optimizer.param_groups[-1]['lr']
        scheduler.step()

        # 记录数据
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # 使用 Logger 输出
        log_msg = (f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        logger.info(log_msg)

        # --- 早停判断 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"--> Best model saved (Val Loss: {avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.warning(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 绘制训练曲线
    plot_training_history(history, save_path)

    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    return model


def evaluate_model(model, test_loader, logger=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
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
    msg = f'Final Evaluation Accuracy: {accuracy:.2f}%'
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return accuracy


def plot_training_history(history, save_path):
    """保存训练曲线图"""
    plt.figure(figsize=(12, 4))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plot_path = os.path.join(os.path.dirname(save_path), 'training_curves.png')
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":

    '''
    ## office 31 real amazon
    batch_size=32
    out_features=31
    num_epochs=20
    save_path = 'checkpoints/Resnet_REAL_amazon/resnet18_amazon.pth'
    train_loader, val_loader, test_loader = prepare_data('original_datasets/office_31/amazon', load_size=128, crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs, save_path=save_path)
    

    ## office 31 real amazon2webcam cyclegan
    batch_size = 32
    out_features = 31
    num_epochs = 20
    save_path = 'checkpoints/Resnet_amazon2webcam_cyclegan/resnet18_amazon2webcam_cyclegan.pth'
    train_loader, val_loader, test_loader = prepare_data('transformed_dataset/cyclegan_128/amazon2webcam', load_size=128, crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs,
                                save_path=save_path)

    ## office 31 real amazon2webcam fgcyclegan
    batch_size = 32
    out_features = 31
    num_epochs = 20
    save_path = 'checkpoints/Resnet_amazon2webcam_fgcyclegan/resnet18_amazon2webcam_fgcyclegan.pth'
    train_loader, val_loader, test_loader = prepare_data('transformed_dataset/fg_cyclegan_128/amazon2webcam',
                                                         load_size=128,
                                                         crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs,
                                save_path=save_path)
    '''

    '''
    ## officehome real art
    batch_size = 32
    out_features = 65
    num_epochs = 20
    save_path = 'checkpoints/Resnet_REAL_art/resnet18_art.pth'
    train_loader, val_loader, test_loader = prepare_data('original_datasets/officehome/Art', load_size=128, crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs, save_path=save_path)
    '''

    ## officehome art2realworld cyclegan
    batch_size = 32
    out_features = 65
    num_epochs = 20
    save_path = 'checkpoints/Resnet_art2realworld_cyclegan/resnet18_art2realworld_cyclegan.pth'
    train_loader, val_loader, test_loader = prepare_data('transformed_dataset/cyclegan_128/art2realword', load_size=128, crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs,
                                save_path=save_path)

    ## officehome art2realworld fgcyclegan
    batch_size = 32
    out_features = 65
    num_epochs = 20
    save_path = 'checkpoints/Resnet_art2realworld_fgcyclegan/resnet18_art2realworld_fgcyclegan.pth'
    train_loader, val_loader, test_loader = prepare_data('transformed_dataset/fg_cyclegan_128/art2realword', load_size=128,
                                                         crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs,
                                save_path=save_path)

    '''
    ## PACS real photo
    batch_size = 32
    out_features = 7
    num_epochs = 20
    save_path = 'checkpoints/Resnet_REAL_photo/resnet18_photo.pth'
    train_loader, val_loader, test_loader = prepare_data('original_datasets/PACS/photo', load_size=128, crop_size=128,
                                                         batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs,
                                save_path=save_path)
    '''

    ## PACS photo2sketch cyclegan
    batch_size = 32
    out_features = 7
    num_epochs = 20
    save_path = 'checkpoints/Resnet_photo2sketch_cyclegan/resnet18_photo2sketch_cyclegan.pth'
    train_loader, val_loader, test_loader = prepare_data('transformed_dataset/cyclegan_128/photo2sketch', load_size=128,crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs,save_path=save_path)

    ## PACS photo2sketch fg_cyclegan
    batch_size = 32
    out_features = 7
    num_epochs = 20
    save_path = 'checkpoints/Resnet_photo2sketch_fgcyclegan/resnet18_photo2sketch_fgcyclegan.pth'
    train_loader, val_loader, test_loader = prepare_data('transformed_dataset/fg_cyclegan_128/photo2sketch', load_size=128,
                                                         crop_size=128, batch_size=batch_size)
    # 训练完成后会得到可以在 CycleGAN 里作为 netCLS 初始化的权重
    trained_model = train_model(train_loader, val_loader, out_features=out_features, num_epochs=num_epochs,
                                save_path=save_path)

