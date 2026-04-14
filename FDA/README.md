# FDA Experiments

先把 source 图像用目标域低频风格做傅里叶替换，生成 FDA source 数据；再用这些数据训练一个 ResNet18 分类器；最后在真实 target test 上报告准确率。

对第一组 MNIST -> USPS：

source_train / val / test = 来自 MNIST
target_train / test = 来自 USPS
FDA 用 USPS train 给 MNIST train/val 注入目标风格，生成 FDA source 数据
再用这些数据训练一个 ResNet18 分类器
最终在 USPS test 上测准确率


这个目录提供一套独立的 `Fourier Domain Adaptation (FDA)` 实验流程，覆盖以下 4 组任务：

1. `MNIST -> USPS`
2. `Amazon -> Webcam` (`Office-31`)
3. `Art -> RealWorld` (`OfficeHome`)
4. `Photo -> Sketch` (`PACS`)

代码目标：

- 使用一套统一的 FDA 核心实现四组数据集
- 自动生成可复现 split
- 导出 FDA 处理后的 source 图像
- 训练分类器并在 target test 上评估
- 支持 `best.pth / last.pth`
- 支持 `--resume` 断点续训
- 保存 `train.log / metrics.json / progress.json`
- 输出 `loss.png / accuracy.png / learning_rate.png`
- 支持 TensorBoard

## 目录结构

```text
code/
├── FDA/
│   ├── __init__.py
│   ├── common.py
│   ├── create_splits.py
│   ├── data.py
│   ├── export_fda.py
│   ├── fourier.py
│   ├── train_fda_classifier.py
│   └── README.md
└── Datasets/
    ├── PACS/
    ├── OfficeHome/
    ├── MNIST2USPS/
    └── office31/
```

你要求的数据目录约定如下：

```text
Datasets/
├── PACS/
│   ├── photo/
│   └── sketch/
├── OfficeHome/
│   ├── Art/
│   └── RealWorld/
├── MNIST2USPS/
│   ├── MNIST/
│   └── usps/
└── office31/
    ├── amazon/
    └── webcam/
```

其中：

- `office31 / OfficeHome / PACS` 支持：
  - `domain/images/class/*.jpg`
  - `domain/class/*.jpg`
- `MNIST2USPS` 复用 `torchvision` 的 `MNIST / USPS` 读取逻辑，并兼容本地已有原始文件

## 输出目录

默认输出会写到：

```text
FDA/outputs/
├── splits/
│   └── <task>/
├── exports/
│   └── <task>/
└── runs/
    └── <task>/
```

其中：

- `splits/<task>/`
  - `source_train.json`
  - `source_val.json`
  - `target_train.json`
  - `target_test.json`
  - `source_test.json` (`MNIST -> USPS` 会有)
  - `classes.json`
  - `meta.json`
  - `stats.json`
- `exports/<task>/`
  - `images/source_train/...`
  - `images/source_val/...`
  - `manifests/source_train.json`
  - `manifests/source_val.json`
  - `export.log`
  - `export_summary.json`
- `runs/<task>/`
  - `best.pth`
  - `last.pth`
  - `train.log`
  - `metrics.json`
  - `progress.json`
  - `loss.png`
  - `accuracy.png`
  - `learning_rate.png`
  - `tensorboard/`

## 任务名

命令里通过 `--task` 指定任务：

- `mnist2usps`
- `office31_a2w`
- `officehome_a2r`
- `pacs_p2s`

## 0. 环境

建议至少安装：

```bash
python -m pip install torch torchvision tensorboard matplotlib pillow numpy
```

## 1. 先生成 split

### MNIST -> USPS

```bash
python -m FDA.create_splits \
  --task mnist2usps \
  --data-root ./Datasets \
  --output-dir ./FDA/outputs/splits/mnist2usps \
  --seed 42
```

### Office-31 Amazon -> Webcam

```bash
python -m FDA.create_splits \
  --task office31_a2w \
  --data-root ./Datasets \
  --output-dir ./FDA/outputs/splits/office31_a2w \
  --source-val-ratio 0.1 \
  --target-test-ratio 0.5 \
  --seed 42
```

### OfficeHome Art -> RealWorld

```bash
python -m FDA.create_splits \
  --task officehome_a2r \
  --data-root ./Datasets \
  --output-dir ./FDA/outputs/splits/officehome_a2r \
  --source-val-ratio 0.1 \
  --target-test-ratio 0.2 \
  --seed 42
```

### PACS Photo -> Sketch

```bash
python -m FDA.create_splits \
  --task pacs_p2s \
  --data-root ./Datasets \
  --output-dir ./FDA/outputs/splits/pacs_p2s \
  --source-val-ratio 0.1 \
  --target-test-ratio 0.2 \
  --seed 42
```

## 2. 导出 FDA 图像

### MNIST -> USPS

```bash
python -m FDA.export_fda \
  --task mnist2usps \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/mnist2usps \
  --output-dir ./FDA/outputs/exports/mnist2usps \
  --beta 0.071 \
  --fda-image-size 28 \
  --seed 42 \
  --skip-existing
```

### Office-31

```bash
python -m FDA.export_fda \
  --task office31_a2w \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/office31_a2w \
  --output-dir ./FDA/outputs/exports/office31_a2w \
  --beta 0.03 \
  --fda-image-size 224 \
  --seed 42 \
  --skip-existing
```

### OfficeHome

```bash
python -m FDA.export_fda \
  --task officehome_a2r \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/officehome_a2r \
  --output-dir ./FDA/outputs/exports/officehome_a2r \
  --beta 0.03 \
  --fda-image-size 224 \
  --seed 42 \
  --skip-existing
```

### PACS

```bash
python -m FDA.export_fda \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --output-dir ./FDA/outputs/exports/pacs_p2s \
  --beta 0.05 \
  --fda-image-size 224 \
  --seed 42 \
  --skip-existing
```

如果你想固定其他条件，只扫描 `beta = 0.01 / 0.02 / 0.03 / 0.05`，可以分别导出：

```bash
python -m FDA.export_fda \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --output-dir ./FDA/outputs/exports/pacs_p2s_b001 \
  --beta 0.01 \
  --fda-image-size 224 \
  --seed 42 \
  --skip-existing

python -m FDA.export_fda \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --output-dir ./FDA/outputs/exports/pacs_p2s_b002 \
  --beta 0.02 \
  --fda-image-size 224 \
  --seed 42 \
  --skip-existing

python -m FDA.export_fda \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --output-dir ./FDA/outputs/exports/pacs_p2s_b003 \
  --beta 0.03 \
  --fda-image-size 224 \
  --seed 42 \
  --skip-existing

python -m FDA.export_fda \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --output-dir ./FDA/outputs/exports/pacs_p2s_b005 \
  --beta 0.05 \
  --fda-image-size 224 \
  --seed 42 \
  --skip-existing
```

说明：

- `--beta` 控制低频替换带宽
- `--skip-existing` 适合导出中断后继续跑
- digits 推荐先试：`0.1 / 0.15 / 0.2`
- RGB 数据集推荐先试：`0.01 / 0.03 / 0.05`

## 3. 训练 FDA 分类器

### MNIST -> USPS

```bash
python -m FDA.train_fda_classifier \
  --task mnist2usps \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/mnist2usps \
  --fda-root ./FDA/outputs/exports/mnist2usps_b0071 \
  --output-dir ./FDA/outputs/runs/mnist2usps_b0071 \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3 \
  --num-workers 4 \
  --image-size 64
```

### Office-31

```bash
python -m FDA.train_fda_classifier \
  --task office31_a2w \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/office31_a2w \
  --fda-root ./FDA/outputs/exports/office31_a2w \
  --output-dir ./FDA/outputs/runs/office31_a2w \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained
```

### OfficeHome

```bash
python -m FDA.train_fda_classifier \
  --task officehome_a2r \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/officehome_a2r \
  --fda-root ./FDA/outputs/exports/officehome_a2r \
  --output-dir ./FDA/outputs/runs/officehome_a2r \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained
```

### PACS

```bash
python -m FDA.train_fda_classifier \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --fda-root ./FDA/outputs/exports/pacs_p2s \
  --output-dir ./FDA/outputs/runs/pacs_p2s \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained
```

如果你想固定训练配置，只扫描不同 `beta` 导出的 FDA 图，可以分别训练：

```bash
python -m FDA.train_fda_classifier \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --fda-root ./FDA/outputs/exports/pacs_p2s_b001 \
  --output-dir ./FDA/outputs/runs/pacs_p2s_b001 \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained

python -m FDA.train_fda_classifier \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --fda-root ./FDA/outputs/exports/pacs_p2s_b002 \
  --output-dir ./FDA/outputs/runs/pacs_p2s_b002 \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained

python -m FDA.train_fda_classifier \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --fda-root ./FDA/outputs/exports/pacs_p2s_b003 \
  --output-dir ./FDA/outputs/runs/pacs_p2s_b003 \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained

python -m FDA.train_fda_classifier \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --fda-root ./FDA/outputs/exports/pacs_p2s_b005 \
  --output-dir ./FDA/outputs/runs/pacs_p2s_b005 \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained
```

如果你想在同样的训练配置下改成更保守的 `Resize + Flip`，可以加上：

```bash
python -m FDA.train_fda_classifier \
  --task pacs_p2s \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/pacs_p2s \
  --fda-root ./FDA/outputs/exports/pacs_p2s_b003 \
  --output-dir ./FDA/outputs/runs/pacs_p2s_b003_resize_flip \
  --epochs 25 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained \
  --train-augment resize_flip
```

说明：

- `--train-augment resize_flip` 对 RGB 数据集使用更保守的 `Resize -> RandomHorizontalFlip`，不再使用 `RandomResizedCrop`
- 推荐先用默认增强扫完 `beta`，再对最优 `beta` 补跑一组 `resize_flip`

## 4. 断点续训

如果训练中断，直接加载 `last.pth` 继续：

```bash
python -m FDA.train_fda_classifier \
  --task office31_a2w \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/office31_a2w \
  --fda-root ./FDA/outputs/exports/office31_a2w \
  --output-dir ./FDA/outputs/runs/office31_a2w \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --pretrained \
  --resume ./FDA/outputs/runs/office31_a2w/last.pth
```

说明：

- `--epochs` 表示总 epoch 数，不是追加 epoch 数
- 例如已经训练到第 8 个 epoch，继续时仍然写 `--epochs 20`

## 5. 只评估已有权重

```bash
python -m FDA.train_fda_classifier \
  --task office31_a2w \
  --data-root ./Datasets \
  --split-dir ./FDA/outputs/splits/office31_a2w \
  --fda-root ./FDA/outputs/exports/office31_a2w \
  --output-dir ./FDA/outputs/runs/office31_a2w \
  --eval-only \
  --resume ./FDA/outputs/runs/office31_a2w/best.pth
```

## 6. TensorBoard

```bash
tensorboard --logdir ./FDA/outputs/runs
```

然后浏览器打开 TensorBoard 显示的地址。

## 7. 推荐实验顺序

建议先跑：

1. `mnist2usps`
2. `office31_a2w`
3. `pacs_p2s`
4. `officehome_a2r`

## 8. 参数建议

- `MNIST -> USPS`
  - `beta`: `0.1 / 0.15 / 0.2`
  - `epochs`: `20`
  - `batch-size`: `128`
- `Office-31`
  - `beta`: `0.01 / 0.03 / 0.05`
  - `epochs`: `20`
  - `pretrained`
- `OfficeHome`
  - `beta`: `0.01 / 0.03 / 0.05`
  - `epochs`: `25`
  - `pretrained`
- `PACS`
  - `beta`: `0.01 / 0.02 / 0.03 / 0.05`
  - `epochs`: `25`
  - `pretrained`
  - 如需更保守增强，可加：`--train-augment resize_flip`

## 9. 结果文件看哪里

- 训练日志：`train.log`
- 导出日志：`export.log`
- 最优权重：`best.pth`
- 最近权重：`last.pth`
- 指标汇总：`metrics.json`
- 训练进度：`progress.json`
- 曲线图：`loss.png / accuracy.png / learning_rate.png`
- TensorBoard：`tensorboard/`

## 10. 注意

- FDA 只使用 `target_train` 作为无标签风格池，不使用 `target_test`
- `OfficeHome` 和 `PACS` 这里使用的是随机可复现 split，不是官方 protocol
- 如果你后面想和你已有的 `source-only / CycleGAN / CyCADA` 严格公平比较，务必让它们共用同一套 split
