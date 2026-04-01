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
#SBATCH --time=4:00:00

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=control_net_train
#SBATCH --output=control_net_train.out
#SBATCH --error=control_net_train.err

### Your script for computation ###
module load anaconda
eval "$(conda shell.bash hook)"
conda activate PYTHON3.10_TORCH2.8
python train.py --dataroot ./datasets/photo2sketch --name photo2sketch --model cycle_gan --load_size 250 --crop_size 227 --batch_size 8 --print_freq 200 --n_epochs 50 --n_epochs_decay 50
