from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import TensorboardCallback, FunctionEncoder

import argparse

from tqdm import trange

# import models
from src.DeepONet import DeepONet
from src.MatrixMethodHelpers import compute_A, train_nonlinear_transformation
from src.SVDEncoder import SVDEncoder
# from src.OperatorEncoder import OperatorEncoder

# import datasets
from src.Datasets.QuadraticSinDataset import QuadraticDataset, SinDataset, plot_source_quadratic, plot_target_sin, plot_transformation_quadratic_sin
from src.Datasets.DerivativeDataset import CubicDerivativeDataset, CubicDataset, plot_source_cubic, plot_target_cubic_derivative, plot_transformation_derivative
from src.Datasets.IntegralDataset import QuadraticIntegralDataset, plot_target_quadratic_integral, plot_transformation_integral
from src.Datasets.MountainCarPoliciesDataset import MountainCarPoliciesDataset, MountainCarEpisodesDataset, plot_source_mountain_car, plot_target_mountain_car, plot_transformation_mountain_car
from src.Datasets.ElasticPlateDataset import ElasticPlateBoudaryForceDataset, ElasticPlateDisplacementDataset,plot_target_boundary, plot_source_boundary_force, plot_transformation_elastic
from src.Datasets.OperatorDataset import CombinedDataset
from src.Datasets.FluidDataset import FluidBoundaryDataset, FluidVelocityDataset, plot_source_distance_to_object, \
    plot_target_fluid_flow, plot_transformation_fluid


def get_dataset(dataset_type:str, test:bool, model_type:str, n_sensors:int):
    # generate datasets
    freeze_example_xs = model_type == "deeponet"  # deeponet has fixed input sensors.
    # NOTE: Most of these datasets are generative, so the data is always unseen, hence no separate test set.
    if dataset_type == "QuadraticSin":
        src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors)
        tgt_dataset = SinDataset(n_examples_per_sample=n_sensors)
    elif dataset_type == "Derivative":
        src_dataset = CubicDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors)
        tgt_dataset = CubicDerivativeDataset(n_examples_per_sample=n_sensors)
    elif dataset_type == "Integral":
        src_dataset = QuadraticDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors)
        tgt_dataset = QuadraticIntegralDataset(n_examples_per_sample=n_sensors)
    elif dataset_type == "MountainCar":
        src_dataset = MountainCarPoliciesDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors)
        tgt_dataset = MountainCarEpisodesDataset(n_examples_per_sample=n_sensors)
    elif dataset_type == "Elastic":
        src_dataset = ElasticPlateBoudaryForceDataset(freeze_example_xs=freeze_example_xs, test=test, n_examples_per_sample=n_sensors)
        tgt_dataset = ElasticPlateDisplacementDataset(test=test, n_examples_per_sample=n_sensors)
    elif dataset_type == "Fluid":
        src_dataset = FluidBoundaryDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors)
        tgt_dataset = FluidVelocityDataset(n_examples_per_sample=n_sensors)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    combined_dataset = CombinedDataset(src_dataset, tgt_dataset, calibration_only=(model_type == "matrix"))

    # sample from all of them to freeze the example inputs, which only matters for deeponet.
    src_dataset.sample("cpu")
    tgt_dataset.sample("cpu")
    combined_dataset.sample("cpu")

    return src_dataset, tgt_dataset, combined_dataset

# test any model on a dataset
def test(model,
         combined_dataset:CombinedDataset,
         callback:TensorboardCallback,
         transformation_type:str,
         train_method:str,
         model_type:str ):
    # set combined dataset to give us more testing data.
    if model_type == "matrix":
        combined_dataset.calibration_only = False

    with torch.no_grad():
        num_trials = 10
        loss = 0
        for test in range(num_trials):
            # Get data
            src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device)

            # Compute y_hats for a given model type
            if model_type == "matrix":
                src_Cs, _ = model["src"].compute_representation(src_xs, src_ys, method=train_method)
                if transformation_type == "linear":
                    tgt_Cs_hat = src_Cs @ model["A"].T
                else:
                    tgt_Cs_hat = model["A"](src_Cs)
                tgt_y_hats = model["tgt"].predict(tgt_xs, tgt_Cs_hat)
            elif model_type == "SVD" or model_type == "Eigen":
                tgt_y_hats = model.predict_from_examples(src_xs, src_ys, tgt_xs, method=train_method, representation_dataset="source", prediction_dataset="target")
            else:
                tgt_y_hats = model.forward(src_xs, src_ys, tgt_xs)

            # Compute loss
            loss += torch.nn.MSELoss()(tgt_y_hats, tgt_ys).item()
        loss = loss / num_trials

    # log under a new tag
    callback.tensorboard.add_scalar("test/mse", loss, callback.total_epochs)

    # Set combined dataset back to training mode for matrix method
    if model_type == "matrix":
        combined_dataset.calibration_only = True



# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--n_sensors", type=int, default=1000)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=10_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_type", type=str, default="SVD")
parser.add_argument("--dataset_type", type=str, default="QuadraticSin")
parser.add_argument("--logdir", type=str, default="logs")
args = parser.parse_args()
assert args.model_type in ["SVD", "Eigen", "matrix", "deeponet"]
assert args.dataset_type in ["QuadraticSin", "Derivative", "Integral", "MountainCar", "Elastic", "Fluid"]


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = args.seed
load_path = args.load_path
model_type = args.model_type
dataset_type = args.dataset_type
nonlinear_datasets = ["MountainCar", "Elastic", "Fluid"]
transformation_type = "nonlinear" if args.dataset_type in nonlinear_datasets else "linear"

print(f"Training {model_type} on {transformation_type} {dataset_type} for {epochs} epochs, seed {seed}, with {n_basis} basis functions and {args.n_sensors} sensors.")

# generate logdir
if load_path is None:
    model_name_for_saving = f"{model_type}_{args.train_method}" if model_type != "deeponet" else model_type
    logdir = f"{args.logdir}/{dataset_type}/{model_name_for_saving}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# generate datasets
src_dataset, tgt_dataset, combined_dataset = get_dataset(dataset_type, test=False, model_type=model_type, n_sensors=args.n_sensors)
_, _, testing_combined_dataset = get_dataset(dataset_type, test=True, model_type=model_type, n_sensors=args.n_sensors)

# if using deeponet, we need to copy the input sensors
if args.model_type == "deeponet":
    testing_combined_dataset.src_dataset.example_xs = combined_dataset.src_dataset.example_xs
    testing_combined_dataset.example_xs = combined_dataset.example_xs
    if dataset_type == "Fluid": # this one specifcally requires us to copy the input sensor indicies.
        testing_combined_dataset.src_dataset.sample_indicies = combined_dataset.src_dataset.sample_indicies

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

    # optionally add neural network to transform between spaces for nonlinear operator
    if transformation_type == "nonlinear":
        a_model = torch.nn.Sequential(
            torch.nn.Linear(src_model.n_basis, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, tgt_model.n_basis),
        ).to(device)
        model["A"] = a_model
        opt = torch.optim.Adam(model["A"].parameters(), lr=1e-3)
    else:
        model["A"] = torch.rand(tgt_model.n_basis, src_model.n_basis).to(device)

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
        if transformation_type == "linear":
            model["A"] = torch.load(f"{logdir}/A.pth")
        else:
            model["A"].load_state_dict(torch.load(f"{logdir}/A.pth"))

    else:
        model.load_state_dict(torch.load(f"{logdir}/model.pth"))
else: # train models
    # create callbacks
    if args.model_type == "matrix":
        callback = TensorboardCallback(logdir=logdir, prefix="source")
        callback2 = TensorboardCallback(tensorboard=callback.tensorboard, prefix="target") # this logs to the same tensorboard but with a different prefix
    else:
        callback = TensorboardCallback(logdir) # this one logs training data

    # train and test occasionally
    test(model, testing_combined_dataset, callback, transformation_type, args.train_method, args.model_type)
    num_tests = 100
    for iteration in trange(num_tests):

        # training step
        if args.model_type == "matrix":
            model["src"].train_model(src_dataset, epochs=epochs//num_tests, callback=callback, progress_bar=False)
            model["tgt"].train_model(tgt_dataset, epochs=epochs//num_tests, callback=callback2, progress_bar=False)
            if transformation_type == "linear":
                model["A"] = compute_A(model["src"], model["tgt"], combined_dataset, device, args.train_method, callback)
            else:
                model["A"], opt = train_nonlinear_transformation(model["A"], opt, model["src"], model["tgt"], args.train_method, combined_dataset, epochs//num_tests, callback)
        else:
            model.train_model(combined_dataset, epochs=epochs//num_tests, callback=callback, progress_bar=False)

        # testing step.
        test(model, testing_combined_dataset, callback, transformation_type, args.train_method, args.model_type)



    # save the model
    if args.model_type == "matrix":
        torch.save(model["src"].state_dict(), f"{logdir}/src_model.pth")
        torch.save(model["tgt"].state_dict(), f"{logdir}/tgt_model.pth")
        if transformation_type == "linear":
            torch.save(model["A"], f"{logdir}/A.pth")
        else:
            torch.save(model["A"].state_dict(), f"{logdir}/A.pth")
    else:
        torch.save(model.state_dict(), f"{logdir}/model.pth")


##############   Evaluate    ###################
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
    elif args.dataset_type == "Elastic":
        plot_source = plot_source_boundary_force
        plot_target = plot_target_boundary
        plot_transformation = plot_transformation_elastic
    elif args.dataset_type == "Fluid":
        plot_source = plot_source_distance_to_object
        plot_target = plot_target_fluid_flow
        plot_transformation = plot_transformation_fluid
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
        elif args.dataset_type == "Fluid":
            x_1 = src_dataset.xx1
            x_2 = src_dataset.xx2
            x_1, x_2 = torch.meshgrid(x_1, x_2)
            xs = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
            xs = xs.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
            ys = src_dataset.ys[info["function_indicies"]]
            xs, ys = xs.to(device), ys.to(device)

        if args.model_type == "matrix":
            y_hats = model["src"].predict_from_examples(example_xs, example_ys, xs, method=args.train_method)
        elif args.model_type == "SVD" or args.model_type == "Eigen":
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="source", prediction_dataset="source")

        # plot source domain
        plot_source(xs, ys, y_hats, info, logdir)

        # get data
        example_xs, example_ys, xs, ys, info = tgt_dataset.sample(device)
        if args.dataset_type == "Fluid":
            x_1 = tgt_dataset.xx1
            x_2 = tgt_dataset.xx2
            x_1, x_2 = torch.meshgrid(x_1, x_2)
            xs = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
            xs = xs.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
            ys = tgt_dataset.ys[info["function_indicies"]]
            xs, ys = xs.to(device), ys.to(device)

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
    elif args.dataset_type == "Fluid":
        x_1 = src_dataset.xx1
        x_2 = src_dataset.xx2
        x_1, x_2 = torch.meshgrid(x_1, x_2)
        grid = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
        grid = grid.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
        grid_outs = src_dataset.ys[info["function_indicies"]]
        grid, grid_outs = grid.to(device), grid_outs.to(device)
        xs = grid
        ys = tgt_dataset.ys[info["function_indicies"]]
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
        if transformation_type == "linear":
            rep = rep @ model["A"].T
        else:
            rep = model["A"](rep)
        y_hats = model["tgt"].predict(xs, rep)
    elif args.model_type == "SVD" or args.model_type == "Eigen":
        y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method, representation_dataset="source", prediction_dataset="target")
    else: # deeponet
        y_hats = model.forward(example_xs, example_ys, xs)

    # plot
    plot_transformation(grid, grid_outs, example_y_hats, xs, ys, y_hats, info, logdir)



