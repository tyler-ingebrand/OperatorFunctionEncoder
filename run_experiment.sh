#!/bin/bash


python test.py --epochs 1000 --model_type "SVD" --dataset_type "QuadraticSin"
python test.py --epochs 1000 --model_type "SVD" --dataset_type "Derivative"
python test.py --epochs 1000 --model_type "SVD" --dataset_type "Integral"
python test.py --epochs 20000 --model_type "SVD" --dataset_type "MountainCar"

python test.py --epochs 1000 --model_type "Eigen" --dataset_type "QuadraticSin"
python test.py --epochs 1000 --model_type "Eigen" --dataset_type "Derivative"
python test.py --epochs 1000 --model_type "Eigen" --dataset_type "Integral"
#python test.py --epochs 20000 --model_type "Eigen" --dataset_type "MountainCar" # This is not self-adjoint, so we cant run eigen

python test.py --epochs 1000 --model_type "matrix" --dataset_type "QuadraticSin"
python test.py --epochs 1000 --model_type "matrix" --dataset_type "Derivative"
python test.py --epochs 1000 --model_type "matrix" --dataset_type "Integral"
python test.py --epochs 20000 --model_type "matrix" --dataset_type "MountainCar"

python test.py --epochs 1000 --model_type "deeponet" --dataset_type "QuadraticSin"
python test.py --epochs 1000 --model_type "deeponet" --dataset_type "Derivative"
python test.py --epochs 1000 --model_type "deeponet" --dataset_type "Integral"
python test.py --epochs 20000 --model_type "deeponet" --dataset_type "MountainCar"