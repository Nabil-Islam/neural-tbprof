#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --job-name="tune_loss"
#SBATCH --account="punim1703"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=1-00:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/tune_loss_%A_%a.out

# One loss-weight setting per array task, 5 CV folds each.
#   sbatch tune_loss.sh
# When every task has finished:
#   python scripts/tune_loss.py er --summarise --tag "$TAG"   (TAG below)
#
# Run from the REVISION root (sbatch from there). Tasks whose
# setting_XX.csv already exists are skipped, so failed tasks can be
# resubmitted with e.g. sbatch --array=3,17 tune_loss.sh

if [ "x$SLURM_JOB_ID" == "x" ]; then
  echo "Submit with sbatch"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=-1
mkdir -p logs

VARIANT_SET="er"
# results go to results/metrics/tuning/${VARIANT_SET}_${TAG}/ ; use a new tag
# whenever the epochs / learning rate change, or finished settings from the
# old run are skipped as "already done"
TAG="ep100"
REF_RUNS="/data/scratch/projects/punim1703/false_negative/old_tbprof/results/essentials/NEW_updated_runs.json"

echo "setting $SLURM_ARRAY_TASK_ID started $(date)"
python3 scripts/tune_loss.py "$VARIANT_SET" \
  --setting "$SLURM_ARRAY_TASK_ID" \
  --tag "$TAG" \
  --ref-runs "$REF_RUNS"
echo "setting $SLURM_ARRAY_TASK_ID finished $(date)"
