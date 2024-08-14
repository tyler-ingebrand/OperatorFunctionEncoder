#!/bin/bash

python download_data.py
sleep 5
./run_experiment.sh
sleep 5
./run_scripts/run_ablation_n_basis.sh
sleep 5
./run_scripts/run_ablation_n_sensors.sh