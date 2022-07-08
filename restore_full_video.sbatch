#!/usr/bin/env bash

#SBATCH --job-name=restore_video
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --qos=long             # Time limit hrs:min:sec
#SBATCH --output=log_restore_video-%j.out    # Standard output and error log
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu

pwd; hostname; date

# Necessary because the default home directory is not writable from the cluster
export HOME="/scratch/neurobiology/zimmer/hannah"

# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Unpack the flag into an actual command
CMD_DIR="/scratch/neurobiology/zimmer/hannah/repos/image_denoising"

CMD="${CMD_DIR}/restore_full_video.py"
FULL_CMD="python $CMD --project_path $project_path"
echo "$FULL_CMD"

if [ "$is_dry_run" ]; then
  echo "Dry run with command: $FULL_CMD"
else
  python "$CMD" --project_path "$project_path"
fi

date
echo "Finished; check logs to determine success"
