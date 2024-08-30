import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import TensorboardCallback, FunctionEncoder, ListCallback

import argparse
from tqdm import trange

from src.Datasets.DarcyDataset import DarcySrcDataset, DarcyTgtDataset, plot_source_darcy, plot_target_darcy, plot_transformation_darcy
from src.Datasets.HeatDataset import HeatSrcDataset, HeatTgtDataset, plot_source_heat, plot_target_heat, plot_transformation_heat
from src.Datasets.L_shapedDataset import LSrcDataset, LTgtDataset, plot_source_L, plot_target_L, plot_transformation_L
from src.DeepONet import DeepONet
from src.DeepONet_CNN import DeepONet_CNN, DeepONet_2Stage_CNN_branch
from src.MatrixMethodHelpers import compute_A, train_nonlinear_transformation, get_num_parameters, get_num_layers, predict_number_params, get_hidden_layer_size, check_parameters
from src.PODDeepONet import DeepONet_POD
from src.SVDEncoder import SVDEncoder

# import datasets
from src.Datasets.QuadraticSinDataset import QuadraticDataset, SinDataset, plot_source_quadratic, plot_target_sin, plot_transformation_quadratic_sin
from src.Datasets.DerivativeDataset import CubicDerivativeDataset, CubicDataset, plot_source_cubic, plot_target_cubic_derivative, plot_transformation_derivative
from src.Datasets.IntegralDataset import QuadraticIntegralDataset, plot_target_quadratic_integral, plot_transformation_integral
from src.Datasets.MountainCarPoliciesDataset import MountainCarPoliciesDataset, MountainCarEpisodesDataset, plot_source_mountain_car, plot_target_mountain_car, plot_transformation_mountain_car
from src.Datasets.ElasticPlateDataset import ElasticPlateBoudaryForceDataset, ElasticPlateDisplacementDataset,plot_target_boundary, plot_source_boundary_force, plot_transformation_elastic
from src.Datasets.OperatorDataset import CombinedDataset

# import grad path
from src.GradPathCallback import GradPathCallback


def get_dataset(dataset_type:str, test:bool, model_type:str, n_sensors:int, device:str, freeze_example_xs:bool=True, **kwargs):
    # generate datasets
    # freeze_example_xs = model_type in ["deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]  # deeponet has fixed input sensors.
    freeze_xs = model_type in ["deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]
    # NOTE: Most of these datasets are generative, so the data is always unseen, hence no separate test set.
    if dataset_type == "QuadraticSin":
        src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = SinDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Derivative":
        src_dataset = CubicDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device, **kwargs)
        tgt_dataset = CubicDerivativeDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device, **kwargs)
    elif dataset_type == "Integral":
        src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = QuadraticIntegralDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "MountainCar":
        src_dataset = MountainCarPoliciesDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = MountainCarEpisodesDataset(n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Elastic":
        src_dataset = ElasticPlateBoudaryForceDataset(freeze_example_xs=freeze_example_xs, test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = ElasticPlateDisplacementDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Darcy":
        src_dataset = DarcySrcDataset(test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = DarcyTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "Heat":
        src_dataset = HeatSrcDataset(test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = HeatTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    elif dataset_type == "LShaped":
        src_dataset = LSrcDataset(test=test, n_examples_per_sample=n_sensors, device=device)
        tgt_dataset = LTgtDataset(test=test, n_examples_per_sample=n_sensors, freeze_xs=freeze_xs, device=device)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    combined_dataset = CombinedDataset(src_dataset, tgt_dataset, calibration_only=(model_type == "matrix"))

    # sample from all of them to freeze the example inputs, which only matters for deeponet.
    src_dataset.sample(device)
    tgt_dataset.sample(device)
    combined_dataset.sample(device)

    return src_dataset, tgt_dataset, combined_dataset


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--n_sensors", type=int, default=1000)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_type", type=str, default="matrix")
parser.add_argument("--dataset_type", type=str, default="Derivative")
parser.add_argument("--logdir", type=str, default="logs_grad_path")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--approximate_number_paramaters", type=int, default=500_000)
parser.add_argument("--unfreeze_sensors", action="store_true")

args = parser.parse_args()
assert args.model_type in ["matrix", "deeponet", ]
assert args.dataset_type in ["QuadraticSin", "Derivative", "Integral", "MountainCar", "Elastic", "Darcy", "Heat", "LShaped"]

# cancel bad combinations
check_parameters(args)

# hyper params
epochs = args.epochs
n_basis = args.n_basis
if args.device == "auto": # automatically choose
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device == "cuda" or args.device == "cpu": # use specificed device
    device = args.device
else: # use cuda device at this index
    device = f"cuda:{int(args.device)}"
seed = args.seed
model_type = args.model_type
dataset_type = args.dataset_type
nonlinear_datasets = ["MountainCar", "Elastic", "Darcy", "Heat", "LShaped"]
transformation_type = "nonlinear" if args.dataset_type in nonlinear_datasets else "linear"
n_layers = args.n_layers
freeze_example_xs = not args.unfreeze_sensors


print(f"Grad path {model_type} on {transformation_type} {dataset_type} for {epochs} epochs, seed {seed}, with {n_basis} basis functions and {args.n_sensors} sensors.")

# generate logdir
model_name_for_saving = f"{model_type}_{args.train_method}" if ("deeponet" not in model_type)else model_type
logdir = f"{args.logdir}/{dataset_type}/{model_name_for_saving}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# seed torch
torch.manual_seed(seed)

# generate datasets
src_dataset, tgt_dataset, combined_dataset = get_dataset(dataset_type, test=False, model_type=model_type, n_sensors=args.n_sensors, device=device, freeze_example_xs=freeze_example_xs)
_, _, testing_combined_dataset = get_dataset(dataset_type, test=True, model_type=model_type, n_sensors=args.n_sensors, device=device,   freeze_example_xs=freeze_example_xs)

# if using deeponet, we need to copy the input sensors
if "deeponet" in args.model_type:
    testing_combined_dataset.src_dataset.example_xs = combined_dataset.src_dataset.example_xs
    testing_combined_dataset.example_xs = combined_dataset.example_xs

# computes the hidden size that most closely reaches the approximate number of parameters, given a number of layers
hidden_size = get_hidden_layer_size(target_n_parameters=args.approximate_number_paramaters,
                                    model_type=model_type,
                                    n_basis=n_basis, n_layers=n_layers,
                                    src_input_space=src_dataset.input_size,
                                    src_output_space=src_dataset.output_size,
                                    tgt_input_space=tgt_dataset.input_size,
                                    tgt_output_space=tgt_dataset.output_size,
                                    transformation_type=transformation_type,
                                    n_sensors=combined_dataset.n_examples_per_sample,
                                    dataset_type=dataset_type,)

# create the model
if args.model_type == "matrix":
    if dataset_type != "Heat":
        src_model = FunctionEncoder(input_size=src_dataset.input_size,
                                    output_size=src_dataset.output_size,
                                    data_type=src_dataset.data_type,
                                    n_basis=n_basis,
                                    method=args.train_method,
                                    model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                    ).to(device)
    else:
        src_model = None # heat dataset has no source space, just temperature and alpha
    tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                                output_size=tgt_dataset.output_size,
                                data_type=tgt_dataset.data_type,
                                n_basis=n_basis+1, # note this makes debugging way easier.
                                method=args.train_method,
                                model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                ).to(device)
    model = {"src": src_model, "tgt": tgt_model}

    # optionally add neural network to transform between spaces for nonlinear operator
    if transformation_type == "nonlinear":
        transformation_input_size = src_model.n_basis if src_model is not None else src_dataset.output_size[0]
        layers = [torch.nn.Linear(transformation_input_size, hidden_size),torch.nn.ReLU()]
        for layer in range(n_layers - 2):
            layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hidden_size, tgt_model.n_basis)]
        a_model = torch.nn.Sequential(*layers).to(device)
        model["A"] = a_model
        opt = torch.optim.Adam(model["A"].parameters(), lr=1e-3)
    else:
        model["A"] = torch.rand(tgt_model.n_basis, src_model.n_basis).to(device)

elif args.model_type == "deeponet":
    model = DeepONet(input_size_tgt=tgt_dataset.input_size[0],
                     output_size_tgt=tgt_dataset.output_size[0],
                     input_size_src=src_dataset.input_size[0],
                     output_size_src=src_dataset.output_size[0],
                     n_input_sensors=combined_dataset.n_examples_per_sample,
                     p=n_basis,
                     n_layers=n_layers,
                     hidden_size=hidden_size,
                     ).to(device)
else:
    raise ValueError(f"Unknown model type: {args.model_type}")

# get number of parameters
n_params = get_num_parameters(model)
predict_n_params = predict_number_params(model_type, combined_dataset.n_examples_per_sample, n_basis, hidden_size, n_layers, src_dataset.input_size, src_dataset.output_size, tgt_dataset.input_size, tgt_dataset.output_size, transformation_type, dataset_type)
assert n_params == predict_n_params, f"Number of parameters is not consistent, expected {predict_n_params}, got {n_params}."

# writes all parameters and saves them
params = {"seed": seed,
          "n_sensors": args.n_sensors,
          "n_basis": n_basis,
          "n_params": n_params,
          "n_layers": n_layers,
          "hidden_size": hidden_size,
          "approximate_number_parameters": args.approximate_number_paramaters,
          "model_type": model_type,
          "train_method": args.train_method,
          "dataset_type": dataset_type,
          "transformation_type": transformation_type,
          "device": device,
          "logdir": logdir,
          "epochs": epochs,
          }
os.makedirs(logdir, exist_ok=True)
torch.save(params, f"{logdir}/params.pth")


# train or load a model
 # train models
# create callbacks
if args.model_type == "matrix":
    callback = GradPathCallback()
    callback2 = GradPathCallback()
else:
    callback = GradPathCallback()


# training step
if args.model_type == "matrix":
    if model["src"] is not None:
        model["src"].train_model(src_dataset, epochs=epochs, callback=callback, progress_bar=True)
    model["tgt"].train_model(tgt_dataset, epochs=epochs, callback=callback2, progress_bar=True)
else:
    model.train_model(combined_dataset, epochs=epochs, callback=callback, progress_bar=True)

# plot the grad path
density = 100
if args.model_type == "matrix":
    callback.plot_grad_path(model["src"], src_dataset, f"grad_path_{dataset_type}_b2b_src.png", density=density)
    callback2.plot_grad_path(model["tgt"], tgt_dataset, f"grad_path_{dataset_type}_b2b_tgt.png", density=density)
else:
    callback.plot_grad_path(model, combined_dataset, f"grad_path_{dataset_type}_deeponet.png", density=density)