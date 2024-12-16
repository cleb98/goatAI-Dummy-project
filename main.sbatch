#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=1-class-coco
#SBATCH --partition=all_usr_prod
#SBATCH --account=tesi_cbellucci
#SBATCH -e slurm-log/%x-%j.err
#SBATCH -o slurm-log/%x-%j.out
# i file di log, .err/.out vengono salvati nella cartella slurm-log con nome <job-name>-<job-id>.err/.out


# Creazione della cartella slurm-log (se non esiste)
mkdir -p slurm-log

# Carica Conda e attiva l'ambiente
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate ptbase

# Esegui lo script Python
python main.py
