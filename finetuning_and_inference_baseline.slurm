#!/bin/bash

#SBATCH --job-name=finetuning-and-inference-baseline
#SBATCH --output=/home/tkoren/mldaw2425/imryziv/slurm-logs/finetuning-and-inference-baseline-%j.out # redirect stdout
#SBATCH --error=/home/tkoren/mldaw2425/imryziv/slurm-logs/finetuning-and-inference-baseline-%j.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=1000
#SBATCH --signal=USR1@120 # gracefully end the job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
export CUDA_VISIBLE_DEVICES=2  # Change to 3 if you want to use GPU 3

export HF_HOME=/home/tkoren/mldaw2425/imryziv/huggingface_cache # Disk quota in regular home is exceeded

# Load Miniconda and activate the environment
echo "Activating Conda environment..."
source /home/tkoren/mldaw2425/imryziv/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source Conda."; exit 1; }
conda activate base || { echo "Failed to activate Conda environment."; exit 1; }

python finetuning.py --sources fpe persuade daigt --base_model distilbert-base-uncased --save_dataset || { echo "Python script failed."; exit 1; }

echo "Job completed successfully!"
