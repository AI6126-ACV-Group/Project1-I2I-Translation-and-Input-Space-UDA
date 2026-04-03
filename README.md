# ACV project info

## Basic Info
- the codes are tested using Python==3.10 Pytorch==2.8 
- codebase: https://github.com/junyanz/CycleGAN

## Getting Started
### Prepare your dataset
- original dataset can be downloaded from the links in ./datasets/datasource
- to prepare dataset for cycleGan, edit the config and run the codes in ./datasets/generate_UDA_datasets.py
- ```python 
   if __name__ == "__main__":
    config = {
        "source_a": "../original_datasets/officehome/Art",  # 请在此处填入domain A 图片的目录
        "source_b": "../original_datasets/officehome/Real World", # 请在此处填入domain B 图片的目录
        "output_root": "art2realworld", # 待生成的数据集文件夹名称
        "split_ratio": 0.8 # 训练集和测试集比例 (当然对于无监督任务而言，训练时只会用到训练集)
    }

    split_uda_dataset(**config)

## CycleGAN train/test

you can use the command in CycleGAN.ipynb to start training and testing

the simplest training command:
-   `python train.py --dataroot dataset_dir --name project_name --model model_name --batch_size batch_size` 

you can find more training options in ./options/base_options.py and ./options/train_options.py

the simplest testing command:
-   `python test.py --dataroot datasets/MINST2USPS/testA --name MINST2USPS --model test --no_dropout`

you can find more training options in ./options/base_options.py and ./options/test_options.py

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

