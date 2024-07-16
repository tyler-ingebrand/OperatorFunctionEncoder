from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import TensorboardCallback, FunctionEncoder

import argparse

# import models
from src.DeepONet import DeepONet
from src.MatrixMethodHelpers import compute_A
from src.SVDEncoder import SVDEncoder
# from src.OperatorEncoder import OperatorEncoder

# import datasets
from src.Datasets.QuadraticSinDataset import QuadraticDataset, SinDataset, plot_source_quadratic, plot_target_sin, plot_transformation_quadratic_sin
from src.Datasets.DerivativeDataset import CubicDerivativeDataset, CubicDataset, plot_source_cubic, plot_target_cubic_derivative, plot_transformation_derivative
from src.Datasets.IntegralDataset import QuadraticIntegralDataset, plot_target_quadratic_integral, plot_transformation_integral
from src.Datasets.MountainCarPoliciesDataset import MountainCarPoliciesDataset, MountainCarEpisodesDataset, plot_source_mountain_car, plot_target_mountain_car, plot_transformation_mountain_car
from src.Datasets.OperatorDataset import CombinedDataset

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=10_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_type", type=str, default="SVD")
parser.add_argument("--dataset_type", type=str, default="QuadraticSin")
args = parser.parse_args()
assert args.model_type in ["SVD", "Eigen", "matrix", "deeponet"]
assert args.dataset_type in ["QuadraticSin", "Derivative", "Integral", "MountainCar"]


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = args.seed
load_path = args.load_path
model_type = args.model_type
dataset_type = args.dataset_type

# generate logdir
if load_path is None:
    model_name_for_saving = f"{model_type}_{args.train_method}" if model_type != "deeponet" else model_type
    logdir = f"logs/{dataset_type}/{model_name_for_saving}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# generate datasets
freeze_example_xs = args.model_type == "deeponet" # deeponet has fixed input sensors.
if dataset_type == "QuadraticSin":
    src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs)
    tgt_dataset = SinDataset()
elif dataset_type == "Derivative":
    src_dataset = CubicDataset(freeze_example_xs=freeze_example_xs)
    tgt_dataset = CubicDerivativeDataset()
elif dataset_type == "Integral":
    src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs)
    tgt_dataset = QuadraticIntegralDataset()
elif dataset_type == "MountainCar":
    src_dataset = MountainCarPoliciesDataset(freeze_example_xs=freeze_example_xs)
    tgt_dataset = MountainCarEpisodesDataset()
else:
    raise ValueError(f"Unknown dataset type: {dataset_type}")
# matrix method only needs data to compute A, so its only for calibration, ie limited datapoints.
combined_dataset = CombinedDataset(src_dataset, tgt_dataset, calibration_only=(args.model_type=="matrix"))

if args.model_type == "Eigen":
    assert src_dataset.input_size == tgt_dataset.input_size, "Eigen can only handle self-adjoint operators, so input sizes must match."
    assert src_dataset.output_size == tgt_dataset.output_size, "Eigen can only handle self-adjoint operators, so output sizes must match."


# create the model
if args.model_type == "SVD" or args.model_type == "Eigen":
    model = SVDEncoder(input_size_src=src_dataset.input_size,
                       output_size_src=src_dataset.output_size,
                       input_size_tgt=tgt_dataset.input_size,
                       output_size_tgt=tgt_dataset.output_size,
                       data_type="deterministic", # we dont support stochastic for now, though its possible.
                       n_basis=n_basis,
                       method=args.train_method,
                       use_eigen_decomp=(model_type=="Eigen")).to(device)
elif args.model_type == "matrix":
    src_model = FunctionEncoder(input_size=src_dataset.input_size,
                                output_size=src_dataset.output_size,
                                data_type=src_dataset.data_type,
                                n_basis=n_basis,
                                method=args.train_method,
                                ).to(device)
    tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                                output_size=tgt_dataset.output_size,
                                data_type=tgt_dataset.data_type,
                                n_basis=n_basis+1, # note this makes debugging way easier.
                                method=args.train_method,
                                ).to(device)
    model = {"src": src_model, "tgt": tgt_model}
else:
    model = DeepONet(input_size_tgt=tgt_dataset.input_size[0],
                     output_size_tgt=tgt_dataset.output_size[0],
                     input_size_src=src_dataset.input_size[0],
                     output_size_src=src_dataset.output_size[0],
                     n_input_sensors=combined_dataset.n_examples_per_sample,
                     p=n_basis,
                     ).to(device)


# train or load a model
if load_path is not None: # load models
    if args.model_type == "matrix":
        model["src"].load_state_dict(torch.load(f"{logdir}/src_model.pth"))
        model["tgt"].load_state_dict(torch.load(f"{logdir}/tgt_model.pth"))
        model["A"] = torch.load(f"{logdir}/A.pth")
    else:
        model.load_state_dict(torch.load(f"{logdir}/model.pth"))
else: # train models
    # create callbacks
    callback = TensorboardCallback(logdir) # this one logs training data

    # train the model
    if args.model_type == "matrix":
        model["src"].train_model(src_dataset, epochs=epochs, callback=callback)
        model["tgt"].train_model(tgt_dataset, epochs=epochs, callback=callback)
        model["A"] = compute_A(model["src"], model["tgt"], combined_dataset, device, args.train_method)
    else:
        model.train_model(combined_dataset, epochs=epochs, callback=callback)

    # save the model
    if args.model_type == "matrix":
        torch.save(model["src"].state_dict(), f"{logdir}/src_model.pth")
        torch.save(model["tgt"].state_dict(), f"{logdir}/tgt_model.pth")
        torch.save(model["A"], f"{logdir}/A.pth")
    else:
        torch.save(model.state_dict(), f"{logdir}/model.pth")



with torch.no_grad():
    # fetch the correct plotting functions
    if args.dataset_type == "QuadraticSin":
        plot_source = plot_source_quadratic
        plot_target = plot_target_sin
        plot_transformation = plot_transformation_quadratic_sin
    elif args.dataset_type == "Derivative":
        plot_source = plot_source_cubic
        plot_target = plot_target_cubic_derivative
        plot_transformation = plot_transformation_derivative
    elif args.dataset_type == "Integral":
        plot_source = plot_source_quadratic
        plot_target = plot_target_quadratic_integral
        plot_transformation = plot_transformation_integral
    elif args.dataset_type == "MountainCar":
        plot_source = plot_source_mountain_car
        plot_target = plot_target_mountain_car
        plot_transformation = plot_transformation_mountain_car
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")


    # plot src and target fit, if using SVD, Eigen, or Matrix
    if args.model_type == "SVD" or args.model_type == "Eigen" or args.model_type == "matrix":
        # get data
        example_xs, example_ys, xs, ys, info = src_dataset.sample(device)
        # mountain car plot needs a 2d grid instead of the random data, for plotting purposes.
        if args.dataset_type == "MountainCar":
            x_1 = torch.linspace(-1.2, 0.6, 100)
            x_2 = torch.linspace(-0.07, 0.07, 100)
            x_1, x_2 = torch.meshgrid(x_1, x_2)
            xs = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
            xs = xs.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
            ys = src_dataset.compute_outputs(info, xs)
            xs, ys = xs.to(device), ys.to(device)

        if args.model_type == "matrix":
            y_hats = model["src"].predict_from_examples(example_xs, example_ys, xs, method=args.train_method)
        elif args.model_type == "SVD" or args.model_type == "Eigen":
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="source", prediction_dataset="source")

        # plot source domain
        plot_source(xs, ys, y_hats, info, logdir)

        # get data
        example_xs, example_ys, xs, ys, info = tgt_dataset.sample(device)
        if args.model_type == "matrix":
            y_hats = model["tgt"].predict_from_examples(example_xs, example_ys, xs, method=args.train_method)
        else:
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="target", prediction_dataset="target")

        # plot target domain
        plot_target(xs, ys, y_hats, info, logdir)


    # plot transformation for all model types
    example_xs, example_ys, xs, ys, info = combined_dataset.sample(device)
    # mountain car plot needs a 2d grid instead of the random data, for plotting purposes.
    if args.dataset_type == "MountainCar":
        x_1 = torch.linspace(-1.2, 0.6, 100)
        x_2 = torch.linspace(-0.07, 0.07, 100)
        x_1, x_2 = torch.meshgrid(x_1, x_2)
        grid = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
        grid = grid.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
        grid_outs = src_dataset.compute_outputs(info, grid)
        grid, grid_outs = grid.to(device), grid_outs.to(device)
    else:
        grid = example_xs
        grid_outs = example_ys


    # first compute example y_hats for the three model types that can do this.
    if args.model_type == "SVD" or args.model_type == "Eigen" or args.model_type == "matrix":
        if args.model_type == "matrix":
            example_y_hats = model["src"].predict_from_examples(example_xs, example_ys, grid, method=args.train_method)
        else:
            example_y_hats = model.predict_from_examples(example_xs, example_ys, grid, method=args.train_method, representation_dataset="source", prediction_dataset="source")
    else:
        example_y_hats = None

    # next compute y_hats for all models
    if args.model_type == "matrix":
        rep, _ = model["src"].compute_representation(example_xs, example_ys, method=args.train_method)
        rep = rep @ model["A"].T
        y_hats = model["tgt"].predict(xs, rep)
    elif args.model_type == "SVD" or args.model_type == "Eigen":
        y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="source", prediction_dataset="target")
    else: # deeponet
        y_hats = model.forward(example_xs, example_ys, xs)

    # plot
    plot_transformation(grid, grid_outs, example_y_hats, xs, ys, y_hats, info, logdir)



