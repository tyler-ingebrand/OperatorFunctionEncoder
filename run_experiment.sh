#!/bin/bash


python test_derivative.py
python test_derivative.py --train_method least_squares

python test_integral.py
python test_integral.py --train_method least_squares