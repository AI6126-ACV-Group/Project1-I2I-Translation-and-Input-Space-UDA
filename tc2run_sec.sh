#!/bin/bash

### TC2 Job Script ###

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
##SBATCH --mem=1G

### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=1

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC2N01

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ###
#SBATCH --time=6:00:00

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=fg_cyclegan
#SBATCH --output=fg_cyclegan.out
#SBATCH --error=fg_cyclegan.err

### Your script for computation ###
module load anaconda
eval "$(conda shell.bash hook)"
conda activate PYTHON3.10_TORCH2.8
python train_cyclegan.py --dataroot ./datasets/photo2sketch --name photo2sketch_fg_cyclegan_128 --model fg_cycle_gan --load_size 150 --crop_size 128 --display_winsize 128 --batch_size 16 --print_freq 200 --n_epochs 200 --n_epochs_decay 0 --D_lr_weight 0.2 --use_wandb --wandb_project_name photo2sketch_fg_cyclegan_128 --wandb_key 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

