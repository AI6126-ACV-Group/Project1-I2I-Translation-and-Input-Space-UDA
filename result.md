# Final Results

| Method | MNIST2USPS | amazon2webcam (128x128) | art2realworld (128x128) | photo2sketch (128x128) |
| --- | ---: | ---: | ---: | ---: |
| source-only | 81.22% | 48.43% | 48.82% | 25.94% |
| cycleGAN | 90.93% | 53.21% | 46.91% | 53.93% |
| cycleGAN + freq_loss | 95.76% | 56.98% | 46.57% | 55.36% |
| cycleGAN + sem_loss | 96.11% | 46.54% | 45.83% | 23.87% |
| cycleGAN + sem_loss with adda | 96.16% | 53.58% | 45.31% | 37.80% |
| cycleGAN + freq_loss with adda | 96.06% | 59.50% | 45.93% | 44.34% |
| FDA | 79.57% | 52.75% | 55.10% | 31.81% |

## Summary

- `MNIST2USPS` 最优结果是 `cycleGAN + sem_loss with adda`，达到 `96.16%`。
- `amazon2webcam` 最优结果是 `cycleGAN + freq_loss with adda`，达到 `59.50%`。
- `art2realworld` 最优结果是 `FDA`，达到 `55.10%`。
- `photo2sketch` 最优结果是 `cycleGAN + freq_loss`，达到 `55.36%`。
