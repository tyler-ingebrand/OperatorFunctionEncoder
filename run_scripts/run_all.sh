#!/bin/bash

# python download_data.py
# sleep 5

# echo "Running main experiment"
# ./run_scripts/run_experiment.sh
# sleep 5

# echo "Running ablation basis"
# ./run_scripts/run_ablation_n_basis.sh
# sleep 5

echo "Running ablation sensors"
./run_scripts/run_ablation_n_sensors.sh
sleep 5

echo "Running ablation unfreeze"
./run_scripts/run_ablation_unfreeze.sh
sleep 5

echo "Running ablation size"
./run_scripts/run_ablation_size.sh
