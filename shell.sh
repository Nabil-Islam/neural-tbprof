#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Feb 18 2025 17:18:14 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=sapphire

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="snp_find"

# The project ID which this job should run under:
#SBATCH --account="punim1703"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

# The amount of memory in megabytes per node:
#SBATCH --mem=53600

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=3-00:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
  echo "You need to submit your job to the queuing system with sbatch"
  exit 1
fi

dir="/data/scratch/projects/punim1703/nn_tb/REVISION/scripts"

for sample in who er; do
  python3 "$dir/make_index.py" "$sample"
  python3 "$dir/weights.py" "$sample"
  python3 "$dir/forward_pass.py" "$sample"
done
python3 "$dir/comp.py"
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
