from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import TensorboardCallback, FunctionEncoder

import argparse
import os
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

# test any model on a dataset
def test(model,
         combined_dataset:CombinedDataset,
         callback:TensorboardCallback,
         transformation_type:str,
         train_method:str,
         model_type:str ):
    # set combined dataset to give us more testing data.
    combined_dataset.calibration_only = False

    with torch.no_grad():
        num_trials = 10
        loss = 0
        for test in range(num_trials):
            # Get data
            src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device)

            # Compute y_hats for a given model type
            src_Cs, _ = model["src"].compute_representation(src_xs, src_ys, method=args.train_method)
            if transformation_type == "linear":
                tgt_Cs_hat = src_Cs @ model["A"].T
            else:
                tgt_Cs_hat = model["A"](src_Cs)
            tgt_y_hats = model["tgt"].predict(tgt_xs, tgt_Cs_hat)
            
            # Compute loss
            loss += torch.nn.MSELoss()(tgt_y_hats, tgt_ys)
        loss = loss / num_trials

    # log under a new tag
    callback.tensorboard.add_scalar("test/mse", loss.item(), callback.total_epochs)

    # Set combined dataset back to training mode for matrix method
    combined_dataset.calibration_only = True



# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--n_sensors", type=int, default=1000)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset_type", type=str, default="Derivative")
parser.add_argument("--logdir", type=str, default="logs_mwe")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--approximate_number_paramaters", type=int, default=500_000)
parser.add_argument("--unfreeze_sensors", action="store_true")

args = parser.parse_args()
assert args.dataset_type in ["Derivative", "Integral",  "Elastic", "Darcy", "LShaped"]

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
model_type = "matrix"
dataset_type = args.dataset_type
nonlinear_datasets = ["Elastic", "Darcy", "Heat", "LShaped"]
transformation_type = "nonlinear" if args.dataset_type in nonlinear_datasets else "linear"
n_layers = args.n_layers
freeze_example_xs = not args.unfreeze_sensors


print(f"Training {model_type} on {transformation_type} {dataset_type} for {epochs} epochs, seed {seed}, with {n_basis} basis functions and {args.n_sensors} sensors.")

# generate logdir
model_name_for_saving = f"{model_type}_{args.train_method}" if ("deeponet" not in model_type)else model_type
logdir = f"{args.logdir}/{dataset_type}/{model_name_for_saving}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


# seed torch
torch.manual_seed(seed)

# generate datasets
src_dataset, tgt_dataset, combined_dataset = get_dataset(dataset_type, test=False, model_type=model_type, n_sensors=args.n_sensors, device=device, freeze_example_xs=freeze_example_xs)
_, _, testing_combined_dataset = get_dataset(dataset_type, test=True, model_type=model_type, n_sensors=args.n_sensors, device=device,   freeze_example_xs=freeze_example_xs)


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

src_model = FunctionEncoder(input_size=src_dataset.input_size,
                                output_size=src_dataset.output_size,
                                data_type=src_dataset.data_type,
                                n_basis=n_basis,
                                method=args.train_method,
                                model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                ).to(device)
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

# train models
# create callbacks
callback = TensorboardCallback(logdir=logdir, prefix="source")
callback2 = TensorboardCallback(tensorboard=callback.tensorboard, prefix="target") # this logs to the same tensorboard but with a different prefix


# train and test occasionally

# we compute A in the linear case, so the initial loss is the best linear operator approximation
# for two random bases
if transformation_type == "linear":
    model["A"] = compute_A(model["src"], model["tgt"], combined_dataset, device, args.train_method, callback)

# test before training
test(model, testing_combined_dataset, callback2, transformation_type, args.train_method, "matrix")

# run training loop
num_tests = 100 if epochs > 0 else 0
for iteration in trange(num_tests):
    # training step
    model["src"].train_model(src_dataset, epochs=epochs//num_tests, callback=callback, progress_bar=False)
    model["tgt"].train_model(tgt_dataset, epochs=epochs//num_tests, callback=callback2, progress_bar=False)
    if transformation_type == "linear":
        model["A"] = compute_A(model["src"], model["tgt"], combined_dataset, device, args.train_method, callback)
    else:
        model["A"], opt, _ = train_nonlinear_transformation(model["A"], opt, model["src"], model["tgt"], args.train_method, combined_dataset, epochs//num_tests, callback, model_type)
    
    # testing step.
    test(model, testing_combined_dataset, callback2, transformation_type, args.train_method, "matrix")




##############   Evaluate    ###################
############## ignore everything below this line, it just makes plots. Go look at logs/ to see the plots.
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
    elif args.dataset_type == "Darcy":
        plot_source = plot_source_darcy
        plot_target = plot_target_darcy
        plot_transformation = plot_transformation_darcy
    elif args.dataset_type == "Heat":
        plot_source = plot_source_heat
        plot_target = plot_target_heat
        plot_transformation = plot_transformation_heat
    elif args.dataset_type == "LShaped":
        plot_source = plot_source_L
        plot_target = plot_target_L
        plot_transformation = plot_transformation_L
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")



    # plot source
    # get data
    example_xs, example_ys, xs, ys, info = src_dataset.sample(device, plot_only=True)
    info["model_type"] = f"matrix_{args.train_method}"

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

    y_hats = model["src"].predict_from_examples(example_xs, example_ys, xs, method=args.train_method)

    # plot source domain
    plot_source(xs, ys, y_hats, info, logdir)

    # get data
    example_xs, example_ys, xs, ys, info = tgt_dataset.sample(device, plot_only=True)
    info["model_type"] = f"matrix_{args.train_method}" 
    if args.dataset_type == "Fluid":
        x_1 = tgt_dataset.xx1
        x_2 = tgt_dataset.xx2
        x_1, x_2 = torch.meshgrid(x_1, x_2)
        xs = torch.stack([x_1.flatten(), x_2.flatten()], dim=1)
        xs = xs.unsqueeze(0).repeat(combined_dataset.n_functions_per_sample, 1, 1)
        ys = tgt_dataset.ys[info["function_indicies"]]
        xs, ys = xs.to(device), ys.to(device)
    elif args.dataset_type == "Heat":
        function_indicies = info["function_indicies"]
        xs = tgt_dataset.xs[function_indicies]
        ys = tgt_dataset.ys[function_indicies]
        times = [0, 20, 40, 60]
        size = 99*99
        new_xs, new_ys = [], []
        for time in times:
            temp_xs = xs[:, size * time: size * (time + 1)]
            temp_ys = ys[:, size * time: size * (time + 1)]
            new_xs.append(temp_xs)
            new_ys.append(temp_ys)
        xs = torch.cat(new_xs, dim=1).to(device)
        ys = torch.cat(new_ys, dim=1).to(device)





    y_hats = model["tgt"].predict_from_examples(example_xs, example_ys, xs, method=args.train_method)

    # plot target domain
    plot_target(xs, ys, y_hats, info, logdir)


    # plot transformation for all model types
    example_xs, example_ys, xs, ys, info = testing_combined_dataset.sample(device, plot_only=True)
    info["model_type"] = f"matrix_{args.train_method}" 

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
    elif args.dataset_type == "Heat":
        function_indicies = info["function_indicies"]
        all_xs = testing_combined_dataset.tgt_dataset.xs[function_indicies]
        all_ys = testing_combined_dataset.tgt_dataset.ys[function_indicies]

        # get subset we want to plot        
        xs = all_xs[:, 49::99, :]
        ys = all_ys[:, 49::99, :]
        grid = example_xs
        grid_outs =  example_ys

    else:
        grid = example_xs
        grid_outs = example_ys


    # first compute example y_hats for the three model types that can do this.
    example_y_hats = model["src"].predict_from_examples(example_xs, example_ys, grid, method=args.train_method)

    # next compute y_hats for all models
    rep, _ = model["src"].compute_representation(example_xs, example_ys, method=args.train_method)
    if transformation_type == "linear":
        rep = rep @ model["A"].T
    else:
        rep = model["A"](rep)
    y_hats = model["tgt"].predict(xs, rep)


    # plot
    if not (args.dataset_type == "Heat" and args.model_type == "deeponet_pod"): # POD cannot be called on new inputs, so it cannot plot.
        plot_transformation(grid, grid_outs, example_y_hats, xs, ys, y_hats, info, logdir)



