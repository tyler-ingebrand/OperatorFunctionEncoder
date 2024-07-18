#!/bin/bash

#python download_flud_data.py

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

#python test.py --epochs 10000 --model_type "SVD" --dataset_type "QuadraticSin"
#python test.py --epochs 10000 --model_type "Eigen" --dataset_type "QuadraticSin"
#python test.py --epochs 10000 --model_type "matrix" --dataset_type "QuadraticSin"
# python test.py --epochs 10000 --model_type "deeponet" --dataset_type "QuadraticSin"

#python test.py --epochs 10000 --model_type "SVD" --dataset_type "Derivative"
#python test.py --epochs 10000 --model_type "Eigen" --dataset_type "Derivative"
#python test.py --epochs 10000 --model_type "matrix" --dataset_type "Derivative"
# python test.py --epochs 10000 --model_type "deeponet" --dataset_type "Derivative"


#python test.py --epochs 10000 --model_type "SVD" --dataset_type "Integral"
#python test.py --epochs 10000 --model_type "Eigen" --dataset_type "Integral"
#python test.py --epochs 10000 --model_type "matrix" --dataset_type "Integral"
# python test.py --epochs 10000 --model_type "deeponet" --dataset_type "Integral"


#python test.py --epochs 20000 --model_type "SVD" --dataset_type "MountainCar" --seed 2
## This is not self-adjoint, so we cant run eigen
#python test.py --epochs 20000 --model_type "matrix" --dataset_type "MountainCar"  --seed 2
# python test.py --epochs 20000 --model_type "deeponet" --dataset_type "MountainCar"  --seed 2

#python test.py --epochs 20000 --model_type "SVD" --dataset_type "Elastic"
## This is not self-adjoint, so we cant run eigen
# python test.py --epochs 20000 --model_type "matrix" --dataset_type "Elastic"
#python test.py --epochs 20000 --model_type "deeponet" --dataset_type "Elastic"