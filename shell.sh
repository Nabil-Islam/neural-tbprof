#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --job-name="snp_find"
#SBATCH --account="punim1703"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=53600
#SBATCH --time=3-00:00:00

# Safety check
if [ "x$SLURM_JOB_ID" == "x" ]; then
  echo "You need to submit your job to the queuing system with sbatch"
  exit 1
fi

# Variables
dir="/data/scratch/projects/punim1703/nn_tb/REVISION/scripts"
res="/data/scratch/projects/punim1703/nn_tb/REVISION/results"

# Define the array (Bash syntax)
samples=("who" "er" "filt")

# The Sequential Loop
for sample in "${samples[@]}"; do
  echo "--------------------------------------------"
  echo "Processing sample: $sample"
  echo "Started at: $(date)"
  echo "--------------------------------------------"

  #python3 "$dir/make_index.py" "$sample"
  #python3 "$dir/weights.py" "$sample"
  #python3 "$dir/forward_pass.py" "$sample"
  python3 "$dir/train_test.py" "$sample"

  echo "Finished $sample at: $(date)"
done

# Run the final comparison once after all samples are done
#echo "All samples complete. Running final comparison..."
python3 "$dir/comp.py"
python3 "$dir/plot.py"
