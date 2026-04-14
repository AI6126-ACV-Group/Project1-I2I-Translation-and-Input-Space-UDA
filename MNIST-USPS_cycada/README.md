## 目录

```text
MNIST-USPS_cycada/
├── train_source_only_resnet18.py
├── train_translated_source_resnet18.py
├── train_cycada_feature_adaptation.py
├── train_digit_cyclegan.py
├── export_translated_digits.py
└── digit_uda/
    ├── __init__.py
    ├── common.py
    └── cyclegan.py
```

## 数据与输出路径

当前 README 默认按下面路径组织：

- 数据根目录：`MNIST-USPS_cycada/data`
- MNIST 数据：`MNIST-USPS_cycada/data/MNIST`
- USPS 数据：`MNIST-USPS_cycada/data/usps`
- 结果目录：`MNIST-USPS_cycada/outputs`

先进入项目目录再运行：

```bash
cd MNIST-USPS_cycada
```

## 0. Source-only baseline

这个 checkpoint 在实验 2 的 `sem_loss` 中会作为语义教师网络使用，也可以当普通 baseline。

```bash
python train_source_only_resnet18.py --source mnist --target usps --data-root ./data --output-dir ./outputs/source_only/mnist_to_usps --epochs 20 --batch-size 128 --lr 1e-3 --num-workers 4
```

## 1. 实验一：CycleGAN + freq_loss

实验一：CycleGAN + freq_loss
用什么数据：
MNIST 和 USPS 的无配对训练图像
之后把 MNIST 翻成 USPS style，生成 translated-source 数据集
在哪个网络：
先在 CycleGAN 上训练图像翻译
再在 ResNet18 分类器上训练 translated-source classifier
做了什么：
训练 freq 版本的 CycleGAN
在普通对抗损失、cycle loss、identity loss 之外，加了 freq_loss
freq_loss 约束的是“循环重建图像”和原图在低频频谱上的一致性
训练好后，用 G_A2B 把 MNIST -> USPS style
用这些 translated-source 图像训练一个 ResNet18
到这里结束，不做 ADDA

### 1.1 训练频域增强 CycleGAN

`freq_loss` 通过约束循环重建图像的低频频谱，增强频域一致性。

```bash
python train_digit_cyclegan.py --variant freq --source mnist --target usps --data-root ./data --output-dir ./outputs/cyclegan_freq/mnist_to_usps --epochs 30 --batch-size 64 --image-size 28 --lr 2e-4 --lambda-cycle 10.0 --lambda-identity 5.0 --lambda-freq 1.0 --low-freq-ratio 0.25 --preview-every 1 --num-workers 4
```

### 1.2 导出 translated-source

```bash
python export_translated_digits.py --source mnist --target usps --data-root ./data --checkpoint ./outputs/cyclegan_freq/mnist_to_usps/best.pth --output-dir ./outputs/translated_digits/mnist_freq_to_usps --zip-output ./outputs/translated_digits/mnist_freq_to_usps.zip --batch-size 128 --image-size 28 --num-workers 4
```

### 1.3 用 translated-source 训练 ResNet

```bash
python train_translated_source_resnet18.py --translated-root ./outputs/translated_digits/mnist_freq_to_usps --target usps --data-root ./data --output-dir ./outputs/translated_source/mnist_freq_to_usps --epochs 20 --batch-size 128 --lr 1e-3 --num-workers 4
```

## 2. 实验二：CycleGAN + sem_loss

用什么数据：
还是 MNIST -> USPS
但这里先额外训练一个 source-only 分类器，作为 semantic teacher
在哪个网络：
先训练一个 source-only ResNet18
再训练 CycleGAN
最后再训练一个 translated-source ResNet18
做了什么：
先用原始 MNIST 训练 source-only ResNet18
把这个分类器冻结，作为 semantic teacher
训练 sem 版本的 CycleGAN
当 G_A2B(MNIST) 生成伪 USPS 图像后，把它送进 teacher
用原来的 MNIST 标签监督 teacher 的输出，要求语义类别别变
然后再导出 translated-source，训练一个 translated-source ResNet18
同样，不做 ADDA

### 2.1 训练 source-only teacher

如果你已经跑过上面的 baseline，可以直接复用 `./outputs/source_only/mnist_to_usps/best.pth`。

```bash
python train_source_only_resnet18.py --source mnist --target usps --data-root ./data --output-dir ./outputs/source_only/mnist_to_usps --epochs 20 --batch-size 128 --lr 1e-3 --num-workers 4
```

### 2.2 训练带语义一致性约束的 CycleGAN

`sem_loss` 会把 `G_A2B(MNIST)` 送进冻结的 source classifier，并用源域标签约束语义不变。

```bash
python train_digit_cyclegan.py --variant sem --source mnist --target usps --data-root ./data --semantic-checkpoint ./outputs/source_only/mnist_to_usps/best.pth --output-dir ./outputs/cyclegan_sem/mnist_to_usps --epochs 30 --batch-size 64 --image-size 28 --lr 2e-4 --lambda-cycle 10.0 --lambda-identity 5.0 --lambda-sem 1.0 --teacher-image-size 224 --preview-every 1 --num-workers 4
```

### 2.3 导出 translated-source

```bash
python export_translated_digits.py --source mnist --target usps --data-root ./data --checkpoint ./outputs/cyclegan_sem/mnist_to_usps/best.pth --output-dir ./outputs/translated_digits/mnist_sem_to_usps --zip-output ./outputs/translated_digits/mnist_sem_to_usps.zip --batch-size 128 --image-size 28 --num-workers 4
```

### 2.4 用 translated-source 训练 ResNet

```bash
python train_translated_source_resnet18.py --translated-root ./outputs/translated_digits/mnist_sem_to_usps --target usps --data-root ./data --output-dir ./outputs/translated_source/mnist_sem_to_usps --epochs 20 --batch-size 128 --lr 1e-3 --num-workers 4
```

## 3. 实验三：CycleGAN + freq_loss with ADDA

用什么数据：
前半段和实验一一样，还是先做 MNIST -> USPS style
后半段额外同时用 translated-source 和真实 USPS
在哪个网络：
先 CycleGAN(freq)
再 translated-source ResNet18
最后做 ADDA / CyCADA-style feature adaptation
source_model: 源模型
target_model: 目标模型
DomainDiscriminator: 域判别器
做了什么：
先训练 freq_loss 版本的 CycleGAN
导出 translated-source 图像
用 translated-source 训练一个 ResNet18，得到 source checkpoint
用这个 checkpoint 初始化 feature adaptation 阶段
在 feature level 上对齐 translated-source 和真实 USPS
通过域判别器区分 source feature / target feature
再训练 target_model.encoder 去“骗过”域判别器
分类头冻结，只调 encoder

### 3.1 训练 `freq_loss` CycleGAN

如果实验一已经跑过，可以直接复用 `./outputs/cyclegan_freq/mnist_to_usps`。

```bash
python train_digit_cyclegan.py --variant freq --source mnist --target usps --data-root ./data --output-dir ./outputs/cyclegan_freq/mnist_to_usps_adda --epochs 30 --batch-size 64 --image-size 28 --lr 2e-4 --lambda-cycle 10.0 --lambda-identity 5.0 --lambda-freq 1.0 --low-freq-ratio 0.25 --preview-every 1 --num-workers 4
```

### 3.2 导出 translated-source

如果你不想重跑 3.1，并且实验一已经产出了 `./outputs/cyclegan_freq/mnist_to_usps/best.pth`，可以直接复用这个生成器：

```bash
python export_translated_digits.py --source mnist --target usps --data-root ./data --checkpoint ./outputs/cyclegan_freq/mnist_to_usps/best.pth --output-dir ./outputs/translated_digits/mnist_freq_to_usps_adda --zip-output ./outputs/translated_digits/mnist_freq_to_usps_adda.zip --batch-size 128 --image-size 28 --num-workers 4
```

如果你按 3.1 重新训练了一个专门给 ADDA 用的 CycleGAN，则使用下面这条命令：

```bash
python export_translated_digits.py --source mnist --target usps --data-root ./data --checkpoint ./outputs/cyclegan_freq/mnist_to_usps_adda/best.pth --output-dir ./outputs/translated_digits/mnist_freq_to_usps_adda --zip-output ./outputs/translated_digits/mnist_freq_to_usps_adda.zip --batch-size 128 --image-size 28 --num-workers 4
```

### 3.3 训练 translated-source ResNet

```bash
python train_translated_source_resnet18.py --translated-root ./outputs/translated_digits/mnist_freq_to_usps_adda --target usps --data-root ./data --output-dir ./outputs/translated_source/mnist_freq_to_usps_adda --epochs 20 --batch-size 128 --lr 1e-3 --num-workers 4
```

### 3.4 用 ADDA 做 feature adaptation

这里直接复用现有的 `train_cycada_feature_adaptation.py`。

```bash
python train_cycada_feature_adaptation.py --translated-root ./outputs/translated_digits/mnist_freq_to_usps_adda --target usps --data-root ./data --source-checkpoint ./outputs/translated_source/mnist_freq_to_usps_adda/best.pth --output-dir ./outputs/cycada_feature/mnist_freq_to_usps_adda --epochs 20 --steps-per-epoch 200 --batch-size 128 --lr-target 1e-6 --lr-discriminator 1e-5 --lambda-adv 0.1 --disc-acc-threshold 0.6 --num-workers 4
```

## 4. 快速调试方法

第一次建议先做 smoke test，确认数据加载、checkpoint 保存和输出目录都正常。

### 4.1 先调 CycleGAN

把 epoch 和 step 都缩小：

```bash
python train_digit_cyclegan.py --variant freq --source mnist --target usps --data-root ./data --output-dir ./outputs/debug/cyclegan_freq --epochs 1 --steps-per-epoch 20 --batch-size 16 --image-size 28 --preview-every 1 --num-workers 0
```

重点检查：

- `./outputs/debug/cyclegan_freq/preview/epoch_001.png`
- `./outputs/debug/cyclegan_freq/metrics.json`
- `./outputs/debug/cyclegan_freq/loss.png`

如果 `preview` 里生成图完全发黑、发白或形状崩坏，先不要继续导出和训练分类器。

### 4.2 再调导出脚本

只导出少量样本：

```bash
python export_translated_digits.py --source mnist --target usps --data-root ./data --checkpoint ./outputs/debug/cyclegan_freq/best.pth --output-dir ./outputs/debug/translated_digits --batch-size 32 --image-size 28 --num-workers 0 --limit 64
```

重点检查：

- 文件名是否形如 `7_000012_fake_B.png`
- 导出数量是否符合预期
- 图像是否能正常打开

### 4.3 最后调 translated-source / ADDA

先把分类器 epoch 缩到 1：

```bash
python train_translated_source_resnet18.py --translated-root ./outputs/debug/translated_digits --target usps --data-root ./data --output-dir ./outputs/debug/translated_source_resnet --epochs 1 --batch-size 32 --lr 1e-3 --num-workers 0
```

ADDA 也先跑 1 个 epoch：

```bash
python train_cycada_feature_adaptation.py --translated-root ./outputs/debug/translated_digits --target usps --data-root ./data --source-checkpoint ./outputs/debug/translated_source_resnet/best.pth --output-dir ./outputs/debug/cycada_feature --epochs 1 --steps-per-epoch 20 --batch-size 32 --lr-target 1e-6 --lr-discriminator 1e-5 --lambda-adv 0.1 --disc-acc-threshold 0.6 --num-workers 0 --diagnostic-target-eval --save-epoch-checkpoints
```

## 5. 输出文件怎么看

### 5.1 CycleGAN 输出

`train_digit_cyclegan.py` 输出目录中常见文件：

- `best.pth`
- `last.pth`
- `metrics.json`
- `loss.png`
- `learning_rate.png`
- `preview/epoch_xxx.png`
- `tensorboard/`

### 5.2 translated-source 分类器输出

- `best.pth`
- `last.pth`
- `metrics.json`
- `loss.png`
- `accuracy.png`
- `learning_rate.png`
- `tensorboard/`

### 5.3 ADDA / feature adaptation 输出

- `best.pth`
- `last.pth`
- `metrics.json`
- `loss.png`
- `accuracy.png`
- `learning_rate.png`
- `tensorboard/`

如果你开启了 `--save-epoch-checkpoints`，还会额外得到 `epoch_001.pth`、`epoch_002.pth` 等逐轮 checkpoint。

## 6. ADDA 调参建议

对 `MNIST -> USPS`，如果 feature adaptation 不稳定，优先按这个顺序调：

1. 把 `--disc-acc-threshold` 从 `0.6` 提高到 `0.65`
2. 把 `--lambda-adv` 从 `0.1` 降到 `0.05`
3. 把 `--steps-per-epoch` 从 `200` 降到 `100`

诊断时重点看 `accuracy.png` 中的：

- `disc_acc`
- `target_update_rate`
- `proxy_val_acc`
- `target_test_acc`
