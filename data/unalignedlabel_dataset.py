import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
from torchvision import datasets, transforms, models

class UnalignedlabelDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # 继承父类的默认选项
        # parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--original_data_dir', type=str, required=True,
                            help='path to the original ImageFolder dataset to get class_to_idx')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        original_resnet_data = datasets.ImageFolder(opt.original_data_dir)
        self.class_to_idx = original_resnet_data.class_to_idx

    def _extract_label_from_path(self, path):
        filename = os.path.basename(path).lower()  # 转小写
        sorted_classes = sorted(self.class_to_idx.keys(), key=len, reverse=True)
        for class_name in sorted_classes:
            if class_name.lower() in filename:  # 统一小写比较
                return self.class_to_idx[class_name]
        raise ValueError(f"无法在文件名 {filename} 中识别出类别...")

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # --- 关键修改 2: 获取并转换标签 ---
        idx_A = self._extract_label_from_path(A_path)
        label_A = torch.tensor(idx_A, dtype=torch.long)
        # 如果 B 域也有标签，同样处理；如果没有，可以赋一个伪标签或 -1
        try:
            idx_B = self._extract_label_from_path(B_path)
            label_B = torch.tensor(idx_B, dtype=torch.long)
        except ValueError:
            label_B = torch.tensor(-1, dtype=torch.long)

        return {
            "A": A,
            "B": B,
            "A_label": label_A,
            "B_label": label_B,
            "A_paths": A_path,
            "B_paths": B_path
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
