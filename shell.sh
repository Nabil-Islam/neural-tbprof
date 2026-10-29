#!/bin/bash
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --job-name="snp_find"
#SBATCH --account="punim1703"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --time=3-00:00:00

# Safety check
if [ "x$SLURM_JOB_ID" == "x" ]; then
  echo "You need to submit your job to the queuing system with sbatch"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=-1

# Variables
dir="/data/scratch/projects/punim1703/nn_tb/REVISION/scripts"
res="/data/scratch/projects/punim1703/nn_tb/REVISION/results"
tbml="/data/scratch/projects/punim1703/nn_tb/REVISION/tbml"

# Define the array (Bash syntax)"who" "er"
samples=("filt")

# The Sequential Loop
for sample in "${samples[@]}"; do
  echo "--------------------------------------------"
  echo "Processing sample: $sample"
  echo "Started at: $(date)"
  echo "--------------------------------------------"

  #python3 "$dir/make_index.py" "$sample"
  python3 "$dir/train_new.py" "$sample"
  #python3 "$dir/new_layer/cr_w.py" "$sample"
  #python3 "$dir/new_layer/cr_fp.py" "$sample"
  #python3 "$dir/train_cr.py" "$sample"

  echo "Finished $sample at: $(date)"
done

#Run the final comparison once after all samples are done
echo "All samples complete. Running final comparison..."
python3 "$dir/comp_new.py"
# python3 "$dir/plot.py"
#python3 "$dir/CR_bfor_aftr.py"
