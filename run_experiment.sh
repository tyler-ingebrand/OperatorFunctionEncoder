#!/bin/bash

# Make a list of the datasets
datasets=("QuadraticSin" "Derivative" "Integral" "MountainCar" "Elastic")
seeds=(1 2 3 4 5 6 7 8 9 10)
algs=("SVD" "Eigen" "matrix" "deeponet")

# loop through the datasets
for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        for alg in "${algs[@]}"
        do
            python test.py --epochs 20000 --model_type $alg --dataset_type $dataset --seed $seed
            sleep 5 # sleep for 5 seconds to avoid memory allocation issues as it takes a second to deallocate memory
        done
    done
done
