#!/bin/sh
#SBATCH --job-name=shikhar # Job name
#SBATCH --ntasks=8 # Run on an eight CPU
#SBATCH --time=1-00:00:00 # Time limit hrs:min:sec
#SBATCH --output=iss%j.out # Standard output and error log
#SBATCH --partition=low_24h_1gpu 
cd ~/Issue-Classifier/
pwd; hostname; date

python src/train.py --DATASET_SUFFIX _codebert --MODEL_NAME codebert --EMB_MODEL_CHECKPOINT microsoft/codebert-base --user shikhar --device gpu