#!/bin/bash

#python download_flud_data.py

# Make a list of the datasets
datasets=("QuadraticSin" "Derivative" "Integral" "MountainCar" "Elastic")
seeds=(1 2 3)
algs=("SVD" "Eigen" "matrix" "deeponet")
n_basis=(100 80 60 40 20 10 5)

# loop through the datasets
for dataset in "${datasets[@]}"
do
    for seed in "${seeds[@]}"
    do
        for alg in "${algs[@]}"
        do
            for n in "${n_basis[@]}"
            do
                python test.py --epochs 20000 --model_type $alg --dataset_type $dataset --seed $seed --n_basis $n --logdir "logs_n_basis"
                sleep 5 # sleep for 5 seconds to avoid memory allocation issues as it takes a second to deallocate memory
            done
        done
    done
done
