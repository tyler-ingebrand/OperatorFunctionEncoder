#!/bin/bash

#python download_flud_data.py

# Make a list of the datasets
datasets=("QuadraticSin" "Derivative" "Integral" "MountainCar" "Elastic")
seeds=(1 2 3)
algs=("SVD" "Eigen" "matrix" "deeponet")
n_sensors=(10 50 100 200 300 400 500 600 700 800 900 1000)

# loop through the datasets
for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        for alg in "${algs[@]}"
        do
            for n in "${n_sensors[@]}"
            do
                python test.py --epochs 20000 --model_type $alg --dataset_type $dataset --seed $seed --n_sensors $n  --logdir "logs_n_sensors"
                sleep 5 # sleep for 5 seconds to avoid memory allocation issues as it takes a second to deallocate memory
            done
        done
    done
done
