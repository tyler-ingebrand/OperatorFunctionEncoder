#!/bin/bash

# Number of GPUs, probably you should use (0 ) which is a single GPU
GPUS=(0 1 3 4 5 6 7) # skip 2 because someone else is using it

# Maximum number of processes per GPU
PROCESSES_PER_GPU=1

# Lock file to manage the queue
LOCK_FILE=/tmp/gpu_lock_file

# Directory to store process status
STATUS_DIR=/tmp/gpu_status

# Create the status directory
mkdir -p $STATUS_DIR

# Initialize the GPU status files
for GPU in "${GPUS[@]}"; do
  echo 0 > $STATUS_DIR/gpu_$GPU
done

# Function to run your ML experiment
run_experiment() {
  SEED=$1
  ALGO=$2
  DATASET=$3
  COUNT=$4
  GPU=$5

  # make a log directory
  LOGDIR="text_logs/$DATASET/$ALGO/$SEED"
  mkdir -p $LOGDIR
  LOGFILE="$LOGDIR/log.txt"

  echo "Running experiment #$COUNT: Dataset=$DATASET, Alg=$ALGO, Seed=$SEED on GPU $GPU"

  # TODO: Replace the following line with your actual experiment command
  # TODO: Ensure your command uses the specified GPU
  python test.py --epochs 30000 --logdir logs_experiment2 --dataset_type $DATASET --model_type $ALGO --seed $SEED --device $GPU > $LOGFILE 2>&1

  # get the exit code, print a warning if bad
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "WARNING: Experiment #$COUNT failed with exit code $EXIT_CODE"
  fi

  # After completion, decrement the GPU process count
  flock $LOCK_FILE bash -c "count=\$(< $STATUS_DIR/gpu_$GPU); echo \$((count - 1)) > $STATUS_DIR/gpu_$GPU"

  # let it deallocate memory
  sleep 1
}

export -f run_experiment
export STATUS_DIR
export LOCK_FILE

# Function to manage the queue and distribute jobs
manage_queue() {
  while read -r JOB; do
    SEED=$(echo $JOB | cut -d ' ' -f 1)
    ALGO=$(echo $JOB | cut -d ' ' -f 2)
    DATASET=$(echo $JOB | cut -d ' ' -f 3)
    COUNT=$(echo $JOB | cut -d ' ' -f 4)

    # Loop to find an available GPU
    while true; do
      for i in "${GPUS[@]}"; do
        # Lock the file and check the GPU status
        if flock $LOCK_FILE bash -c "[ \$(< $STATUS_DIR/gpu_$i) -lt $PROCESSES_PER_GPU ]"; then
          # Update the GPU status and start the experiment
          flock $LOCK_FILE bash -c "count=\$(< $STATUS_DIR/gpu_$i); echo \$((count + 1)) > $STATUS_DIR/gpu_$i"
          run_experiment $SEED $ALGO $DATASET $COUNT $i &
          break 2
        fi
      done
      # Wait before retrying to find an available GPU
      sleep 2
    done
    # Wait before starting the next job
    sleep 2
  done
  wait
}

# TODO: Generate the job list
job_list=()
count=0
for dataset in "Derivative" "Integral" "Elastic" "Darcy" "Heat" "LShaped"; do
  for algo in "SVD" "Eigen" "matrix" "deeponet" "deeponet_cnn" "deeponet_pod" "deeponet_2stage" "deeponet_2stage_cnn"; do
    for seed in {1..10}; do
      job_list+=("$seed $algo $dataset $count")
      count=$((count + 1))
    done
  done
done



# Convert job list to a format suitable for the manage_queue function
printf "%s\n" "${job_list[@]}" | manage_queue