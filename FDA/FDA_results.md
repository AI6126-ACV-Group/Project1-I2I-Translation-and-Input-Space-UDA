# FDA Results

## 1. MNIST -> USPS

当前 FDA 实现采用的是: 一张 `MNIST` 随机配一张 `USPS`，用目标图的低频频谱替换源图。

### 最终采用参数

- `beta = 0.071`
- `fda-image-size = 28`
- `train image-size = 64`
- `epochs = 20`
- `batch-size = 128`
- `lr = 1e-3`

### 最终结果

- Best checkpoint: `./FDA/outputs/runs/mnist2usps_b0071/best.pth`
- Last checkpoint: `./FDA/outputs/runs/mnist2usps_b0071/last.pth`
- Source test loss: `0.01611075924637262`
- Source test accuracy: `99.62%`
- FDA validation accuracy: `99.55%`
- Target test accuracy: `79.57%`


### 参数对比

| Setting | FDA val acc | Target test acc | Source test acc |
| --- | ---: | ---: | ---: |
| `beta > 0.1` | about `99.6%` | `13.15%` |  |
| `beta = 0.036`, `fda-image-size = 28`, `image-size = 64` | `99.60%` | `76.83%` | - |
| `beta = 0.071`, `fda-image-size = 28`, `image-size = 64` | `99.55%` | `79.57%` | `99.62%` |
| `beta = 0.036`, `fda-image-size = 56`, `image-size = 64` | `99.40%` | `79.32%` | `99.49%` |

### 结论

1. `beta > 0.1` 时，`Target test accuracy` 会掉到接近随机猜测，说明 FDA 频谱替换过强，digits 语义被明显污染。
2. 将 `beta` 调小到 `0.036 ~ 0.071` 后，`Target test accuracy` 从 `13.71%` 提升到 `76% ~ 79%`，说明调小 `beta` 是有效的。
3. `source_test_acc` 仍能保持在 `99.5%+`，说明最终参数下语义基本保住了，问题不再是数字被改坏，而是 FDA 图像与真实 `USPS` 之间仍存在风格分布差异。
4. `FDA validation accuracy` 始终接近 `99.5%`，但不能代表真实目标域泛化能力，因为该指标只在 FDA 导出的 `source_val` 上评估。
5. 把 `fda-image-size` 从 `28` 提高到 `56` 没有带来明显收益，当前瓶颈更像是随机单目标配对带来的结构泄漏和伪影，而不是 FFT 工作分辨率本身。
6. 在 `MNIST -> USPS` 上，这个纯 FDA translated-source 方案最终仍未超过 `source-only` baseline，因此相比之下，`CycleGAN + sem_loss / CyCADA` 更适合这个任务。

## 2. 用对比图论证调小 beta 有效

为了直接展示 `beta` 调整前后的差异，可以生成四列对比图:

1. 原始 `MNIST`
2. 大 `beta` 的 FDA 图
3. 最终参数 `beta = 0.071` 的 FDA 图
4. 对应的 `USPS` 参考图

已添加脚本: `FDA/generate_beta_comparisons.py`

### 生成 4 组类别对比图

如果大 `beta` 的导出目录是 `./FDA/outputs/exports/mnist2usps`，最终参数对应的导出目录是 `./FDA/outputs/exports/mnist2usps_b0071`，可以运行:

```bash
python -m FDA.generate_beta_comparisons \
  --data-root ./Datasets \
  --bad-export-root ./FDA/outputs/exports/mnist2usps \
  --good-export-root ./FDA/outputs/exports/mnist2usps_b0071 \
  --output-dir ./FDA/outputs/figures/mnist2usps_beta_compare_0_3 \
  --classes 0,1,2,3 \
  --samples-per-class 1 \
  --bad-label "FDA beta>0.1" \
  --good-label "FDA beta=0.071"
```

### 再生成另外 4 组类别对比图

```bash
python -m FDA.generate_beta_comparisons \
  --data-root ./Datasets \
  --bad-export-root ./FDA/outputs/exports/mnist2usps \
  --good-export-root ./FDA/outputs/exports/mnist2usps_b0071 \
  --output-dir ./FDA/outputs/figures/mnist2usps_beta_compare_4_7 \
  --classes 4,5,6,7 \
  --samples-per-class 1 \
  --bad-label "FDA beta>0.1" \
  --good-label "FDA beta=0.071"
```

### 生成最后 2 组类别对比图

```bash
python -m FDA.generate_beta_comparisons \
  --data-root ./Datasets \
  --bad-export-root ./FDA/outputs/exports/mnist2usps \
  --good-export-root ./FDA/outputs/exports/mnist2usps_b0071 \
  --output-dir ./FDA/outputs/figures/mnist2usps_beta_compare_8_9 \
  --classes 8,9 \
  --samples-per-class 1 \
  --bad-label "FDA beta>0.1" \
  --good-label "FDA beta=0.071"
```

每次运行后会输出:

- 多张单样本对比图: `01_class*.png`
- 一张总览图: `overview.png`
- 一个记录所选样本的清单: `selection.json`

### 看图时重点观察

1. 大 `beta` 版本是否把 `USPS` 参考图中的结构和背景伪影明显带进了 `MNIST`。
2. `beta = 0.071` 版本是否明显减轻了背景块状纹理和数字形状泄漏。
3. 调小 `beta` 后，数字本体是否更接近原始 `MNIST`，同时又保留部分目标域灰度风格。


amazon->webcam
Best checkpoint: ./FDA/outputs/runs/office31_a2w/best.pth
Last checkpoint: ./FDA/outputs/runs/office31_a2w/last.pth
FDA validation accuracy: 85.16%
Target test accuracy: 52.75%

art -> real world (officehome)
Best checkpoint: ./FDA/outputs/runs/officehome_a2r/best.pth
Last checkpoint: ./FDA/outputs/runs/officehome_a2r/last.pth
FDA validation accuracy: 57.55%
Target test accuracy: 55.10%


photo -> sketch
Best checkpoint: ./FDA/outputs/runs/pacs_p2s/best.pth
Last checkpoint: ./FDA/outputs/runs/pacs_p2s/last.pth
FDA validation accuracy: 46.11%
Target test accuracy: 27.99%