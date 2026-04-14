# Office-31 CyCADA (Amazon -> Webcam)

这个目录提供了一套可直接复用的 `Office-31 Amazon -> Webcam` 实验骨架，目标是完成：

1. `Source-only`
2. `CycleGAN approach`
3. `CyCADA feature adaptation`

下游分类器统一使用 `ResNet18`，并支持：

- `best.pth / last.pth`
- `--resume` 断点续训
- `train.log`
- `metrics.json / progress.json`
- `tensorboard`
- `loss.png / accuracy.png / learning_rate.png`
- 翻译结果预览图

## 先看这个：上传到 AutoDL 需要哪些代码？目录怎么放？

你至少需要准备两部分代码：

1. 你的实验代码目录 `office31_cycada/`
2. 外部图像翻译仓库 `pytorch-CycleGAN-and-pix2pix/`

其中：

- `office31_cycada/` 负责：
  - 生成 Office-31 split
  - 准备 CycleGAN 数据目录
  - 导出翻译后的 `amazon2webcam`
  - 训练 `Source-only`
  - 训练 `CycleGAN baseline classifier`
  - 训练 `CyCADA feature adaptation`

- `pytorch-CycleGAN-and-pix2pix/` 负责：
  - 训练 CycleGAN
  - 用训练好的生成器把 Amazon 图像翻译成 Webcam 风格

### 最少需要上传的内容

```text
/root/
├── office31_cycada/
│   ├── office31_uda/
│   │   ├── __init__.py
│   │   └── common.py
│   ├── prepare_office31_splits.py
│   ├── prepare_cyclegan_office31.py
│   ├── export_amazon2webcam.py
│   ├── train_source_only_resnet18.py
│   ├── train_translated_source_resnet18.py
│   └── train_cycada_feature_adaptation.py
├── pytorch-CycleGAN-and-pix2pix/
├── office31/
│   ├── amazon/
│   │   └── images/
│   │       ├── backpack/
│   │       ├── bike/
│   │       └── ...
│   └── webcam/
│       └── images/
│           ├── backpack/
│           ├── bike/
│           └── ...
└── outputs/
    ├── office31_splits/
    ├── cyclegan_datasets/
    ├── cyclegan_checkpoints/
    ├── cyclegan_results/
    ├── office31_translated/
    └── office31_outputs/
```

如果你本地没有改过 `pytorch-CycleGAN-and-pix2pix/`，通常不需要手动上传整个仓库，推荐直接在 AutoDL 上 `git clone`。

### 运行后会自动生成的目录

```text
/root/outputs/
├── office31_splits/
├── cyclegan_datasets/
├── cyclegan_checkpoints/
├── cyclegan_results/
├── office31_translated/
└── office31_outputs/
```

## 目录说明

- `office31_uda/common.py`
  - 通用工具、数据集、`ResNet18`、判别器、日志、checkpoint、曲线可视化
- `prepare_office31_splits.py`
  - 生成 `amazon_train / amazon_val / webcam_train / webcam_test`
- `prepare_cyclegan_office31.py`
  - 把 split 清单整理成 CycleGAN 需要的 `trainA/trainB/testA/testB`
- `export_amazon2webcam.py`
  - 把 CycleGAN 输出整理成 `amazon2webcam/images/<class>/*.png`
- `train_source_only_resnet18.py`
  - `amazon -> webcam` 的 source-only baseline
- `train_translated_source_resnet18.py`
  - `amazon2webcam -> webcam` 的 translated-source baseline
- `train_cycada_feature_adaptation.py`
  - 用 `amazon2webcam + webcam(unlabeled)` 做 feature adaptation

## 数据集结构

默认假设你的 Office-31 数据是下面这种结构：

```text
/root/office31/
├── amazon/
│   └── images/
│       ├── backpack/
│       ├── bike/
│       └── ...
└── webcam/
    └── images/
        ├── backpack/
        ├── bike/
        └── ...
```

如果你的目录是：

```text
/root/office31/amazon/backpack/*.jpg
```

也可以，脚本会自动兼容 `domain/images` 和 `domain` 两种层级。

下面 README 里的示例命令统一按你的当前放法书写：

- 数据集根目录：`/root/office31`
- 项目目录：`/root/office31_cycada`
- 中间输出目录：统一放在 `/root/outputs/`

## AutoDL 额外准备（建议在第 0 步之前做）

如果你还没有 `pytorch-CycleGAN-and-pix2pix/`，建议在开始任何实验前先在 AutoDL 上拉取；最晚也要在执行 `1.2 训练 CycleGAN` 之前完成。
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/archive/refs/heads/master.zip

```bash
cd /root
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

拉取后，按外部仓库自己的 README 安装依赖；如果仓库内提供了 `requirements.txt`，通常可以直接执行：

```bash
cd /root/pytorch-CycleGAN-and-pix2pix
python -m pip install dominate visdom wandb
```

另外，这个 `office31_cycada/` 目录自己的脚本至少依赖：

- `torch`
- `torchvision`
- `numpy`
- `Pillow`

如果你希望正常输出训练曲线和 TensorBoard，再额外确认环境里有：

- `matplotlib`
- `tensorboard`

## 0. 先生成可复现 split

```bash
cd /root/office31_cycada

python prepare_office31_splits.py \
  --data-root /root/office31 \
  --output-dir /root/outputs/office31_splits/amazon_to_webcam \
  --source-val-ratio 0.1 \
  --target-test-ratio 0.5 \
  --seed 42
```

生成后你会得到：

- `amazon_train.json`
- `amazon_val.json`
- `webcam_train.json`
- `webcam_test.json`
- `classes.json`
- `meta.json`
- `stats.json`

## 1. 你现在要先跑的 CycleGAN 部分

你提到的流程是：

- 阶段 A：`amazon + webcam -> 训练 CycleGAN -> 得到 G_A2W`
- 阶段 B：`amazon --G_A2W--> amazon2webcam`

这两步对应的命令如下。

### 1.1 准备 CycleGAN 数据目录

```bash
cd /root/office31_cycada

python prepare_cyclegan_office31.py \
  --data-root /root/office31 \
  --output-dir /root/outputs/cyclegan_datasets/amazon2webcam \
  --source-manifests /root/outputs/office31_splits/amazon_to_webcam/amazon_train.json /root/outputs/office31_splits/amazon_to_webcam/amazon_val.json \
  --target-manifests /root/outputs/office31_splits/amazon_to_webcam/webcam_train.json \
  --copy-mode copy \
  --limit-test-b 128
```

这一步会生成：

```text
/root/outputs/cyclegan_datasets/amazon2webcam/
├── trainA/
├── trainB/
├── testA/
├── testB/
└── metadata/
    ├── trainA_index.json
    ├── trainB_index.json
    ├── testA_index.json
    └── testB_index.json
```

其中：

- `trainA` = Amazon 源图
- `trainB` = Webcam 训练图
- `testA` = 需要被翻译成 `amazon2webcam` 的 Amazon 图像

### 1.2 训练 CycleGAN

下面命令假设你已经在 AutoDL 上准备好了标准 `pytorch-CycleGAN-and-pix2pix` 仓库；如果还没有，请先执行上面的 `git clone` 和依赖安装步骤。

```bash
cd /root/pytorch-CycleGAN-and-pix2pix

python train.py \
  --dataroot /root/outputs/cyclegan_datasets/amazon2webcam \
  --name amazon2webcam_cyclegan \
  --model cycle_gan \
  --direction AtoB \
  --dataset_mode unaligned \
  --input_nc 3 \
  --output_nc 3 \
  --load_size 286 \
  --crop_size 256 \
  --preprocess resize_and_crop \
  --batch_size 1 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --save_epoch_freq 5 \
  --checkpoints_dir /root/outputs/cyclegan_checkpoints
```

输出重点看：

- `/root/outputs/cyclegan_checkpoints/amazon2webcam_cyclegan/latest_net_G_A.pth`
- `/root/outputs/cyclegan_checkpoints/amazon2webcam_cyclegan/latest_net_G_B.pth`

这里 `G_A` 对应 `Amazon -> Webcam-style`。

### 1.3 用训练好的 G_A2W 翻译整个 Amazon 集合

```bash
cd /root/pytorch-CycleGAN-and-pix2pix

export OMP_NUM_THREADS=1

python test.py \
  --dataroot /root/outputs/cyclegan_datasets/amazon2webcam \
  --name amazon2webcam_cyclegan \
  --model cycle_gan \
  --direction AtoB \
  --dataset_mode unaligned \
  --phase test \
  --num_test 1000000 \
  --load_size 128 \
  --crop_size 128 \
  --preprocess resize \
  --no_dropout \
  --checkpoints_dir /root/outputs/cyclegan_checkpoints \
  --results_dir /root/outputs/cyclegan_results
```

翻译结果通常会出现在：

```text
/root/outputs/cyclegan_results/amazon2webcam_cyclegan/test_latest/images
```

### 1.4 导出成 `amazon2webcam` 分类训练集

```bash
cd /root/office31_cycada

python export_amazon2webcam.py \
  --cycle-index /root/outputs/cyclegan_datasets/amazon2webcam/metadata/testA_index.json \
  --cyclegan-images-dir /root/outputs/cyclegan_results/amazon2webcam_cyclegan/test_latest/images \
  --translated-root /root/outputs/office31_translated/amazon2webcam \
  --preview-count 12 \
  --seed 42
```

导出后会得到：

```text
/root/outputs/office31_translated/amazon2webcam/
├── images/
│   ├── backpack/
│   ├── bike/
│   └── ...
├── manifests/
│   ├── amazon_train.json
│   ├── amazon_val.json
│   └── all.json
├── previews/
│   └── translation_preview.png
└── export_meta.json
```

## 2. 训练 Source-only baseline

```bash
cd /root/office31_cycada

python train_source_only_resnet18.py \
  --data-root /root/office31 \
  --split-dir /root/outputs/office31_splits/amazon_to_webcam \
  --output-dir /root/outputs/office31_outputs/source_only \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --image-size 224 \
  --pretrained
```

断点继续：

```bash
python train_source_only_resnet18.py \
  --data-root /root/office31 \
  --split-dir /root/outputs/office31_splits/amazon_to_webcam \
  --output-dir /root/outputs/office31_outputs/source_only \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --image-size 224 \
  --pretrained \
  --resume /root/outputs/office31_outputs/source_only/last.pth
```

## 3. 训练 CycleGAN baseline 分类器

```bash
cd /root/office31_cycada

python train_translated_source_resnet18.py \
  --translated-root /root/outputs/office31_translated/amazon2webcam \
  --split-dir /root/outputs/office31_splits/amazon_to_webcam \
  --output-dir /root/outputs/office31_outputs/translated_source \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --image-size 224 \
  --pretrained
```

## 4. 训练 CyCADA feature adaptation

```bash
cd /root/office31_cycada

python train_cycada_feature_adaptation.py \
  --translated-root /root/outputs/office31_translated/amazon2webcam \
  --split-dir /root/outputs/office31_splits/amazon_to_webcam \
  --source-checkpoint /root/outputs/office31_outputs/translated_source/best.pth \
  --output-dir /root/outputs/office31_outputs/cycada_feature \
  --epochs 30 \
  --steps-per-epoch 100 \
  --batch-size 32 \
  --lr-target 1e-6 \
  --lr-discriminator 1e-5 \
  --lambda-adv 0.3 \
  --disc-acc-threshold 0.6 \
  --num-workers 4 \
  --image-size 224 \
  --pretrained
```

如果你要先做“训练策略诊断”，建议先额外打开这两个选项：

```bash
cd /root/office31_cycada

python train_cycada_feature_adaptation.py \
  --translated-root /root/outputs/office31_translated/amazon2webcam \
  --split-dir /root/outputs/office31_splits/amazon_to_webcam \
  --source-checkpoint /root/outputs/office31_outputs/translated_source/best.pth \
  --output-dir /root/outputs/office31_outputs/cycada_feature_diag \
  --epochs 30 \
  --steps-per-epoch 100 \
  --batch-size 32 \
  --lr-target 1e-6 \
  --lr-discriminator 1e-5 \
  --lambda-adv 0.3 \
  --disc-acc-threshold 0.6 \
  --num-workers 4 \
  --image-size 224 \
  --pretrained \
  --diagnostic-target-eval \
  --save-epoch-checkpoints
```

说明：

- `--disc-acc-threshold 0.6`：只有当 discriminator 准确率大于 `0.6` 时，才更新 target encoder。
- `--lr-target 1e-6`：让 target encoder 更新更慢，先验证是不是训练太激进导致不稳定。
- `--lambda-adv 0.3`：先减弱 adversarial loss，避免 target encoder 为了骗过 discriminator 而破坏分类边界。
- `--diagnostic-target-eval`：每个 epoch 额外在 `target_test` 上评估一次，只用于诊断“`best.pth` 是否选错”，不要把这次运行直接作为正式汇报结果。
- `--save-epoch-checkpoints`：额外保存每个 epoch 的 checkpoint，便于你回看某个具体 epoch。

实验记录：
epoch 18 的红线”target_test_acc“最高，你可以这样理解：

当前训练策略是有效的，target_test_acc 在中后期达到最好
但 proxy_val_acc 更早达到高点
所以当前用 proxy_val_acc 选 best.pth 会偏早，存在选模偏差

断点继续：

```bash
python train_cycada_feature_adaptation.py \
  --translated-root /root/outputs/office31_translated/amazon2webcam \
  --split-dir /root/outputs/office31_splits/amazon_to_webcam \
  --source-checkpoint /root/outputs/office31_outputs/translated_source/best.pth \
  --output-dir /root/outputs/office31_outputs/cycada_feature \
  --epochs 30 \
  --steps-per-epoch 100 \
  --batch-size 32 \
  --lr-target 1e-6 \
  --lr-discriminator 1e-5 \
  --lambda-adv 0.3 \
  --disc-acc-threshold 0.6 \
  --num-workers 4 \
  --image-size 224 \
  --pretrained \
  --resume /root/outputs/office31_outputs/cycada_feature/last.pth
```

## 5. 输出文件怎么看

每个训练目录里会有：

- `best.pth`
- `last.pth`
- `train.log`
- `progress.json`
- `metrics.json`
- `loss.png`
- `accuracy.png`
- `learning_rate.png`
- `tensorboard/`

如果你开启了 `--save-epoch-checkpoints`，还会多出：

- `epoch_001.pth`
- `epoch_002.pth`
- ...

如果服务器断联：

1. 重新连接后找到 `last.pth`
2. 用同一个命令加上 `--resume last.pth`
3. 训练会从上一个保存 epoch 的下一轮继续

### 5.1 怎么判断 `best.pth` 可能选错了

不要只靠肉眼看一眼图，而是按下面这套标准判断：

1. 先跑一次带 `--diagnostic-target-eval` 的诊断实验。
2. 打开输出目录里的 `accuracy.png`，看 `proxy_val_acc` 和 `target_test_acc` 两条曲线是否在同一个 epoch 附近达到峰值。
3. 再看 `metrics.json` 或 `progress.json` 里的 `history`，确认峰值对应的 epoch 编号。

如果出现下面任一情况，就说明当前用 `proxy_val_acc` 选 `best.pth` 很可能不理想：

- `proxy_val_acc` 的最高点对应的 epoch，不是 `target_test_acc` 的最高点。
- `proxy_val_acc` 后面开始下降，但 `target_test_acc` 还在上升。
- `best.pth` 的最终 `target_test_acc` 明显低于某个中间 epoch 或低于 `last.pth`。

更具体地说：

- 如果两条曲线走势基本同步，只是有轻微噪声，那么当前 proxy 还算能用。
- 如果两条曲线明显错峰，例如 `proxy_val_acc` 在第 8 个 epoch 最高，但 `target_test_acc` 在第 16 个 epoch 才最高，那么你现在的 `best.pth` 就更像是“最适合 translated validation”的模型，而不是“最适合真实 webcam”的模型。

注意：

- `target_test_acc` 这条曲线只用于诊断训练策略，不应用来做正式模型选择。
- 正式实验里，如果你确认 proxy 失效，下一步应该改选模方法，而不是直接拿 `target_test` 选最佳 epoch。

### 5.2 诊断实验跑完后怎么做决定

建议按下面的顺序判断：

1. 先看 `target_update_rate` 是否长期接近 `0`。
如果接近 `0`，说明 `disc_acc > 0.6` 很少成立，门控过严，可以把阈值降到 `0.55`。

2. 再看 `disc_acc` 是否长期很高，例如一直在 `0.8` 以上。
如果一直很高，说明 target encoder 没有成功迷惑 discriminator，可以把 `lambda_adv` 从 `0.3` 调到 `0.5` 再试。

3. 看 `proxy_val_acc` 和 `target_test_acc` 是否错峰。
如果明显错峰，说明主要问题是选模，不一定是训练完全失败。

4. 看 `translated_source -> CyCADA` 是否提升。
如果从 `translated_source` 的 target accuracy 到诊断 run 的最佳 target accuracy 有提升，说明 feature adaptation 是有效的，只是未必选到了最优 checkpoint。

## 6. 推荐实验顺序

1. 上传 `office31_cycada/`，并准备好 `/root/office31/`
2. 在 AutoDL 上拉取 `pytorch-CycleGAN-and-pix2pix/` 并安装依赖
3. `prepare_office31_splits.py`
4. `prepare_cyclegan_office31.py`
5. 跑 CycleGAN `train.py`
6. 跑 CycleGAN `test.py`
7. `export_amazon2webcam.py`
8. `train_source_only_resnet18.py`
9. `train_translated_source_resnet18.py`
10. `train_cycada_feature_adaptation.py`

如果你已经跑完前面流程，只想验证“是不是训练策略问题”，那么这次只需要重跑第 10 步。

更具体地说：

1. 不需要重跑 CycleGAN 训练。
2. 不需要重跑 `export_amazon2webcam.py`。
3. 不需要重跑 `source-only` baseline。
4. 不需要重跑 `translated_source` baseline。
5. 先重跑一次 `train_cycada_feature_adaptation.py`，建议开 `--diagnostic-target-eval` 和 `--save-epoch-checkpoints`。
6. 如果诊断确认这些超参更稳，再用同样超参重跑一次正式版 `train_cycada_feature_adaptation.py`，去掉 `--diagnostic-target-eval`。

## 7. 备注

- 这套代码默认目标域训练只使用 `webcam_train.json`，不使用目标测试标签。
- `proxy_val_acc` 是在 `amazon2webcam_val` 上评估的代理指标，用来选 `CyCADA` 的最佳 checkpoint。
- 当前 `train_cycada_feature_adaptation.py` 已加入 `disc_acc` 门控、较小的 `lr_target` 默认值，以及较弱的 `lambda_adv` 默认值，优先用于验证训练稳定性问题。
- 如果你后面还要做报告里的可视化，优先使用：
  - `previews/translation_preview.png`
  - 训练曲线 PNG
  - `metrics.json` 里的准确率结果
