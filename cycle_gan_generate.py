import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from models import networks  # 假设你的 networks.py 在 models 目录下
import numpy as np
from options.train_options import TrainOptions

# --- 1. 定义生成专用的预处理 (去掉随机性) ---
class TranslationTransform:
    def __init__(self, load_size):
        self.load_size = load_size

    def __call__(self, img):
        # 1. 保持比例缩放 (Zy 修改后的逻辑)
        w, h = img.size
        if w < h:
            new_w = self.load_size
            new_h = self.load_size * h // w
        else:
            new_h = self.load_size
            new_w = self.load_size * w // h
        # 2. 第一次缩放：此时图片至少有一边是 128，另一边大于或等于 128
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # 3. 中心裁剪：从中间切出一个 128x128 的正方形
        # 这样物体的主体特征保持了原始比例，没有被压扁
        left = (new_w - self.load_size) / 2
        top = (new_h - self.load_size) / 2
        right = (new_w + self.load_size) / 2
        bottom = (new_h + self.load_size) / 2
        img = img.crop((left, top, right, bottom))
        # 4. 标准化
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return t(img)

# 将 Tensor 转回图片保存 ---
def tensor_to_pil(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
    return Image.fromarray(image_numpy)

def generate_fake_dataset(data_dir, model_path, output_dir, load_size=128, gpu_ids=[0]):
    device = torch.device(f'cuda:{gpu_ids[0]}') if torch.cuda.is_available() else torch.device('cpu')

    # 3. 初始化并加载生成器 G_A (将 A 映射到 B) 这里我偷懒了，netG_A 参数需要和 训练cycle gan时一致  可以查看 option 中设置
    netG_A = networks.define_G(3, 3, 64, "resnet_9blocks", "instance",
                               False, "normal", 0.02)

    state_dict = torch.load(model_path, map_location=device)
    if 'netG_A' in state_dict:
        netG_A.load_state_dict(state_dict['netG_A'])
    else:
        netG_A.load_state_dict(state_dict)
    netG_A.to(device)

    netG_A.eval()
    print(f"Model loaded from {model_path}")

    # 4. 准备数据集 (使用 ImageFolder 以保持结构)
    transform = TranslationTransform(load_size)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # shuffle 必须为 False，以便我们手动获取路径
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 5. 开始遍历并保存
    print(f"Starting translation: {data_dir} -> {output_dir}")
    for i, (real_A, _) in enumerate(loader):
        # 获取原始文件路径和类别名
        orig_path, _ = dataset.samples[i]
        rel_path = os.path.relpath(orig_path, data_dir)  # 获取如 "back_pack/0001.jpg"
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with torch.no_grad():
            fake_B = netG_A(real_A.to(device))
            result_img = tensor_to_pil(fake_B)
            result_img.save(save_path)
        if i % 100 == 0:
            print(f"Processed {i} images...")
    print("Task finished!")

if __name__ == "__main__":
    # 配置你的路径
    #SRC_DIR = './original_datasets/office_31/amazon'
    #CHECKPOINT = './checkpoints/amazon2webcam_cyclegan_128/latest_net_G_A.pth'
    #DEST_DIR = './transformed_dataset/cyclegan_128/amazon2webcam'

    #SRC_DIR = './original_datasets/officehome/Art'
    #CHECKPOINT = './checkpoints/art2realworld_cyclegan_128/latest_net_G_A.pth'
    #DEST_DIR = './transformed_dataset/cyclegan_128/art2realword'

    SRC_DIR = './original_datasets/PACS/photo'
    CHECKPOINT = './checkpoints/photo2sketch_fg_cyclegan_128/latest_net_G_A.pth'
    DEST_DIR = './transformed_dataset/fg_cyclegan_128/photo2sketch'

    generate_fake_dataset(SRC_DIR, CHECKPOINT, DEST_DIR)