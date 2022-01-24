#!/bin/bash

# Job name:
#SBATCH --job-name=bnn_hmc_sampling
#
# Project:
#SBATCH --account=nn9284k
#
# Wall time limit:
#SBATCH --time=00-00:15:00
#
# Other parameters:
#SBATCH --ntasks=1

#SBATCH --mem-per-cpu=4G

#SBATCH --partition=accel

#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default

source tf_env_default/bin/activate
module load TensorFlow/2.4.1-fosscuda-2020b
pip3 install tensorflow_probability==0.12.1 tqdm

python3 train_models.py --train --gpu -p "(1000022, 1000022)" -f "models/1000022_1000022_hmc.npz" --epochs 1000 --burn 0 --kernel hmc --skip 10 --results 10 --chains 1 > log.txt
