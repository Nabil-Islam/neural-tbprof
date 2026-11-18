#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --job-name="train_all"
#SBATCH --account="punim1703"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/train_%A_%a.out

# Final training with the tuned loss settings, one variant set per task:
#   task 0 = who, 1 = er, 2 = filt
#   sbatch train_all.sh                 # all three
#   sbatch --array=0,1 train_all.sh     # skip filt until its variant set is rebuilt
# Run from the REVISION root. Then compare with TB-Profiler on the test isolates:
#   sbatch --dependency=afterok:<jobid> --wrap="python3 scripts/comp_new.py" ...
#   (or just run comp_new.py via sbatch once all three have finished)

if [ "x$SLURM_JOB_ID" == "x" ]; then
  echo "Submit with sbatch"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=-1
export MPLBACKEND=Agg
mkdir -p logs

SETS=(who er filt)
SET=${SETS[$SLURM_ARRAY_TASK_ID]}
TUNED_SET="er_ep100"     # tuning run whose best_settings.json is used
REF_RUNS="/data/scratch/projects/punim1703/false_negative/old_tbprof/results/essentials/NEW_updated_runs.json"

echo "training $SET with settings from tuning/$TUNED_SET, started $(date)"
python3 scripts/train_new.py "$SET" \
  --tuned --tuned-set "$TUNED_SET" \
  --ref-runs "$REF_RUNS"
echo "finished $SET $(date)"
