#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --job-name="train_cr"
#SBATCH --account="punim1703"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/train_cr_%A_%a.out

# Cross-resistance model, same protocol as train_all.sh:
#   task 0 = who, 1 = er   (filt waits for its rebuilt variant set)
#   sbatch train_cr.sh
# Needs models/naiive_models/<name>_CR_naiive_model.h5 for each set.
# Then rerun comp_new.py: it picks up results/metrics/cr/*_CR_trained_calls.csv.

if [ "x$SLURM_JOB_ID" == "x" ]; then
  echo "Submit with sbatch"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=-1
export MPLBACKEND=Agg
mkdir -p logs

SETS=(who er filt)
SET=${SETS[$SLURM_ARRAY_TASK_ID]}
TUNED_SET="er_ep100"
REF_RUNS="/data/scratch/projects/punim1703/false_negative/old_tbprof/results/essentials/NEW_updated_runs.json"

echo "training CR model on $SET, started $(date)"
python3 scripts/train_cr_new.py "$SET" \
  --tuned --tuned-set "$TUNED_SET" \
  --ref-runs "$REF_RUNS"
echo "finished $SET $(date)"
