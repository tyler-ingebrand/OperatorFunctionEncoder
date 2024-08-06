#!/bin/bash

python download_fluid_data.py
./run_experiment.sh
./run_ablation_n_basis.sh
./run_ablation_n_sensors.sh