#!/bin/sh

#SBATCH --job-name=imryziv-create-datasets
#SBATCH --output=/home/tkoren/mldaworkshop/imryziv/imryziv-create-datasets-%j.out # redirect stdout
#SBATCH --error=/home/tkoren/mldaworkshop/imryziv/imryziv-create-datasets-%j.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=1000
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

source /home/tkoren/mldaw2425/imryziv/Detect_AI_Generated_Text/venv/bin/activate

python





