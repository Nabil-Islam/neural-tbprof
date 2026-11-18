#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --job-name="diag_epochs"
#SBATCH --account="punim1703"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/diag_epochs_%A_%a.out

# Epoch / learning-rate diagnostic: the current loss (setting 0) on CV fold 0,
# trained long enough to see where the validation loss bottoms out.
#   task 0: lr 1e-3, up to 300 epochs  -> results/metrics/tuning/er_diag_lr1e-3/
#   task 1: lr 1e-2, up to 150 epochs  -> results/metrics/tuning/er_diag_lr1e-2/
# Submit from the REVISION root:  sbatch diag_epochs.sh

if [ "x$SLURM_JOB_ID" == "x" ]; then
  echo "Submit with sbatch"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=-1
mkdir -p logs

REF_RUNS="/data/scratch/projects/punim1703/false_negative/old_tbprof/results/essentials/NEW_updated_runs.json"

LRS=(1e-3 1e-2)
EPOCHS=(300 150)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
EP=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

echo "lr $LR, max epochs $EP, started $(date)"
python3 scripts/tune_loss.py er --setting 0 --folds 0 \
  --max-epochs "$EP" --lr "$LR" --tag "diag_lr$LR" \
  --ref-runs "$REF_RUNS"
echo "finished $(date)"
