#!/bin/bash

# Job name:
#SBATCH --job-name=bnn_training
#
# Project:
#SBATCH --account=nn9284k
#
# Wall time limit:
#SBATCH --time=00-72:00:00
#
# Other parameters:
#SBATCH --ntasks=1

#SBATCH --mem-per-cpu=16G

#SBATCH --partition=accel

#SBATCH --gpus=1


## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default

source tf_env/bin/activate
module load cuDNN/8.0.4.30-CUDA-11.1.1 Python/3.8.6-GCCcore-10.2.0
pip3 install tensorflow==2.7.0 tensorflow_probability==0.15.0 tqdm pandas 

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!

for epoch in 0 32 64 128 256 512 1024 2048 4096 8192 
do
	python3 timing_nuts.py --gpu -p "(1000022, 1000022)" --epochs ${epoch} --batch 32 --burn 1000 --kernel hmc --results 1000 --chains 1 --skip 10 --arch "[5, 20, 20, 1]" --act "tanh" --leapfrog 512
done
kill -SIGINT "$NVIDIA_MONITOR_PID"
