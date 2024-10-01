from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_specs import colors, labels, titles

from FunctionEncoder import TensorboardCallback, FunctionEncoder
plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

import argparse
import os
from tqdm import trange
import numpy as np
import copy
import matplotlib
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


def loss_fe(model, example_xs, example_ys, xs, ys):
    # approximate functions, compute error
    representation, gram = model.compute_representation(example_xs, example_ys, method=model.method)
    y_hats = model.predict(xs, representation, precomputed_average_ys=None)
    prediction_loss = model._distance(y_hats, ys, squared=True).mean()

    # LS requires regularization since it does not care about the scale of basis
    # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
    # from going to infinity.
    if model.method == "least_squares":
        norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()

    # add loss components
    loss = prediction_loss
    if model.method == "least_squares":
        loss = loss + model.regularization_parameter * norm_loss

    return loss

def loss_deeponet(model, example_xs, example_ys, xs, ys):
    # sample input data
    xs, u_xs, ys, G_u_ys = example_xs, example_ys, xs, ys

    # approximate functions, compute error
    estimated_G_u_ys = model.forward(xs, u_xs, ys)
    prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)
    return prediction_loss

def get_gradients(model, loss_fn, example_xs, example_ys, xs, ys):
    # zero the gradients
    model.opt.zero_grad()

    # compute loss
    loss = loss_fn(model, example_xs, example_ys, xs, ys)

    # compute gradients
    loss.backward()

    # get gradients
    gradients = []
    for p in model.parameters():
        gradients.append(p.grad.detach().clone())
    return gradients

def get_set_of_gradients(model, loss_fn, dataset, n_functions_to_grad):
    n_loops = n_functions_to_grad // dataset.n_functions_per_sample
    gradients = []
    for _ in range(n_loops):
        example_xs, example_ys, xs, ys, _ = dataset.sample(None)
        for i in range(example_xs.shape[0]):
            f_grads = get_gradients(model, loss_fn, example_xs[i:i+1], example_ys[i:i+1], xs[i:i+1], ys[i:i+1])
            gradients.append(f_grads)
    return gradients

def pca_gradients(set_of_gradients):
    # stack the gradients
    flattened_gradients = []
    for gradients in set_of_gradients:
        flattened_gradients.append(torch.cat([g.flatten() for g in gradients]))

    # stack the gradients
    stacked_gradients = torch.stack(flattened_gradients)

    # compute the mean
    means = stacked_gradients.mean(dim=0, keepdim=True)

    # subtract the mean
    stacked_gradients = stacked_gradients - means

    # do svd on the centered data
    U, S, V = torch.svd(stacked_gradients)

    # get the principal components
    principal_components = V[:, :2]
    return principal_components



def unflatten_principal_components(principal_components, model):
    unflattened_principal_components = []
    start = 0
    for p in model.parameters():
        size = p.numel()
        unflattened_principal_components.append(principal_components[start:start+size, :].reshape(*p.shape, 2))
        start += size
    return unflattened_principal_components

def get_principal_directions(model, loss_fn, dataset, n_functions_to_grad):
    set_of_gradients = get_set_of_gradients(model, loss_fn, dataset, n_functions_to_grad)
    princ_components = pca_gradients(set_of_gradients)
    principal_components = unflatten_principal_components(princ_components, model)
    principal_components = normalize_based_on_filters(model, principal_components)
    return principal_components

def normalize_based_on_filters(model, principal_components):
    principal_components = [pc / torch.norm(pc, dim=(0,1)) for pc in principal_components]
    norms = [torch.norm(p) for p in model.parameters()]
    principal_components = [pc * n for pc, n in zip(principal_components, norms)]
    return principal_components

        

def plot_loss_landscape(ax, model, loss_fn, dataset, n_functions_to_grad, principal_components, density, range=(-1,1)):
    # going to do \theta = \theta* + \alpha * v_1 + \beta * v_2
    alphas = np.linspace(range[0], range[1], density)
    betas = np.linspace(range[0], range[1], density)

    # sample data
    example_xs, example_ys, xs, ys, _ = dataset.sample(None)

    # create a grid of losses
    losses = np.zeros((density, density))
    p_bar = trange(density**2)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # get the new model
            new_model = copy.deepcopy(model)
            for p, pc in zip(new_model.parameters(), principal_components):
                p.data = p.data + alpha * pc[..., 0] + beta * pc[..., 1]

            # compute the loss
            losses[i, j] = loss_fn(new_model, example_xs, example_ys, xs, ys).item()

            del new_model
            p_bar.update(1)
    vmin = np.min(losses)
    vmax = vmin * 1e3
    vmin, vmax = np.log(vmin), np.log(vmax)

    # plot the loss landscape in 3d
    X, Y = np.meshgrid(alphas, betas)
    losses = np.log(losses)
    ax.plot_surface(X, Y, losses, cmap='coolwarm', antialiased=False, vmin=vmin, vmax=vmax,) #  edgecolor='none', linewidth=0,)
    ax.plot_wireframe(X, Y, losses, color='black', alpha=0.5, rstride=11, cstride=11)
    
    # ax.set_xlabel('alpha')
    # ax.set_ylabel('beta')
    # ax.set_zlabel('loss')

    # disable the axis, we just want the image
    ax.axis('off')

    # add color bar
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', shrink=0.5)
    # set labels to normal scale
    cbar.set_ticks([vmin, vmin + np.log(10.0), vmin + np.log(100.0), vmax])
    cbar.set_ticklabels([f"{np.exp(vmin):.2e}", f"{np.exp(vmin + np.log(10.0)):.2e}", f"{np.exp(vmin + np.log(100.0)):.2e}", f"{np.exp(vmax):.2e}"])



    return ax, cbar

if __name__ == "__main__":

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_functions_to_grad', type=int, default=100)
    parser.add_argument('--density', type=int, default=121)

    parser.add_argument("--n_basis", type=int, default=100)
    parser.add_argument("--n_sensors", type=int, default=1000)
    parser.add_argument("--train_method", type=str, default="least_squares")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_type", type=str, default="Derivative")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--approximate_number_paramaters", type=int, default=500_000)
    parser.add_argument("--unfreeze_sensors", action="store_true")
    args = parser.parse_args()


    assert args.dataset_type in ["QuadraticSin", "Derivative", "Integral", "MountainCar", "Elastic", "Darcy", "Heat", "LShaped"]

    # hyper params
    n_basis = args.n_basis
    if args.device == "auto": # automatically choose
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" or args.device == "cpu": # use specificed device
        device = args.device
    else: # use cuda device at this index
        device = f"cuda:{int(args.device)}"
    seed = args.seed
    dataset_type = args.dataset_type
    nonlinear_datasets = ["MountainCar", "Elastic", "Darcy", "Heat", "LShaped"]
    transformation_type = "nonlinear" if args.dataset_type in nonlinear_datasets else "linear"
    n_layers = args.n_layers
    freeze_example_xs = not args.unfreeze_sensors

    # load dir
    exp_dir = "logs_experiment"
    load_dir = f"{exp_dir}/{dataset_type}/"
    load_dir_deeponet = f"{load_dir}/deeponet/"
    load_dir_matrix = f"{load_dir}/matrix_least_squares/"

    # get all subdirs for both
    subdirs_deeponet = [d for d in os.listdir(load_dir_deeponet) if os.path.isdir(os.path.join(load_dir_deeponet, d))]
    subdirs_matrix = [d for d in os.listdir(load_dir_matrix) if os.path.isdir(os.path.join(load_dir_matrix, d))]

    # organize them by date
    subdirs_deeponet = sorted(subdirs_deeponet, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))
    subdirs_matrix = sorted(subdirs_matrix, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))
    # for s in subdirs_matrix:
    #     print(s)

    # load dirs
    b2b_load_path = f"{load_dir_matrix}/{subdirs_matrix[seed-1]}"
    deeponet_load_path = f"{load_dir_deeponet}/{subdirs_deeponet[seed-1]}"
    ab_range = (-0.01, 0.01)

    # seed torch
    torch.manual_seed(seed)


    # create the dataset
    src_dataset, tgt_dataset, combined_dataset = get_dataset(dataset_type, test=False, model_type="matrix", n_sensors=args.n_sensors, device=device, freeze_example_xs=freeze_example_xs)

    # get hidden sizes
    deeponet_hidden_size = get_hidden_layer_size(target_n_parameters=args.approximate_number_paramaters,
                                        model_type="deeponet",
                                        n_basis=n_basis, n_layers=n_layers,
                                        src_input_space=src_dataset.input_size,
                                        src_output_space=src_dataset.output_size,
                                        tgt_input_space=tgt_dataset.input_size,
                                        tgt_output_space=tgt_dataset.output_size,
                                        transformation_type=transformation_type,
                                        n_sensors=combined_dataset.n_examples_per_sample,
                                        dataset_type=dataset_type,)

    matrix_hidden_size = get_hidden_layer_size(target_n_parameters=args.approximate_number_paramaters,
                                        model_type="matrix",
                                        n_basis=n_basis, n_layers=n_layers,
                                        src_input_space=src_dataset.input_size,
                                        src_output_space=src_dataset.output_size,
                                        tgt_input_space=tgt_dataset.input_size,
                                        tgt_output_space=tgt_dataset.output_size,
                                        transformation_type=transformation_type,
                                        n_sensors=combined_dataset.n_examples_per_sample,
                                        dataset_type=dataset_type,)
    # create the models

    # matrix method
    if dataset_type != "Heat":
        src_model = FunctionEncoder(input_size=src_dataset.input_size,
                                    output_size=src_dataset.output_size,
                                    data_type=src_dataset.data_type,
                                    n_basis=n_basis,
                                    method=args.train_method,
                                    model_kwargs={"n_layers":n_layers, "hidden_size":matrix_hidden_size},
                                    ).to(device)
    else:
        src_model = None # heat dataset has no source space, just temperature and alpha
    tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                                output_size=tgt_dataset.output_size,
                                data_type=tgt_dataset.data_type,
                                n_basis=n_basis+1, # note this makes debugging way easier.
                                method=args.train_method,
                                model_kwargs={"n_layers":n_layers, "hidden_size":matrix_hidden_size},
                                ).to(device)
    matrix_model = {"src": src_model, "tgt": tgt_model}

    # deepONet
    deeponet_model = DeepONet(input_size_tgt=tgt_dataset.input_size[0],
                    output_size_tgt=tgt_dataset.output_size[0],
                    input_size_src=src_dataset.input_size[0],
                    output_size_src=src_dataset.output_size[0],
                    n_input_sensors=combined_dataset.n_examples_per_sample,
                    p=n_basis,
                    n_layers=n_layers,
                    hidden_size=deeponet_hidden_size,
                    ).to(device)
    
    # load the models
    # matrix
    if matrix_model["src"] is not None:
        matrix_model["src"].load_state_dict(torch.load(f"{b2b_load_path}/src_model.pth", weights_only=True))
    matrix_model["tgt"].load_state_dict(torch.load(f"{b2b_load_path}/tgt_model.pth", weights_only=True))

    # deeponet
    deeponet_model.load_state_dict(torch.load(f"{deeponet_load_path}/model.pth", weights_only=True))


    # create the axis, we need 3 plots
    fig = plt.figure(figsize=(15, 5), dpi=500)
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # first plot matrix method src in ax1
    pad = +30
    principal_components = get_principal_directions(matrix_model["src"], loss_fe, src_dataset, args.n_functions_to_grad)
    ax1, cbar1 = plot_loss_landscape(ax1, matrix_model["src"], loss_fe, src_dataset, args.n_functions_to_grad, principal_components, args.density, range=ab_range)
    ax1.set_title("Decomposed B2B Input \n Function Space Loss",  y=0.0, pad=pad)
    # rotate the plot
    ax1.view_init(elev=30, azim=20)

    # second plot matrix method tgt in ax2
    principal_components = get_principal_directions(matrix_model["tgt"], loss_fe, tgt_dataset, args.n_functions_to_grad)
    ax2, cbar2 = plot_loss_landscape(ax2, matrix_model["tgt"], loss_fe, tgt_dataset, args.n_functions_to_grad, principal_components, args.density, range=ab_range)
    ax2.set_title("Decomposed B2B Output \n Function Space Loss",  y=0.0, pad=pad)
    ax2.view_init(elev=30, azim=-60)

    # third plot deepONet in ax3
    principal_components = get_principal_directions(deeponet_model, loss_deeponet, combined_dataset, args.n_functions_to_grad)
    ax3, cbar3 = plot_loss_landscape(ax3, deeponet_model, loss_deeponet, combined_dataset, args.n_functions_to_grad, principal_components, args.density, range=ab_range)
    ax3.set_title("End-To-End \n DeepONet Loss",  y=0.0, pad=pad)
    ax3.view_init(elev=30, azim=30)

    # add line between ax2 and ax3
    left = cbar2.ax.get_position().xmax + 0.041
    right = ax3.get_position().xmin
    xpos = (left+right) / 2
    top = cbar2.ax.get_position().ymax + 0.11
    bottom = cbar2.ax.get_position().ymin - 0.11
    line1 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top),transform=fig.transFigure, color="black", linestyle="--", lw=2)
    fig.lines = line1, 


    # now save it
    plt.tight_layout()
    plt.savefig(f"loss_landscape_{dataset_type}.png", bbox_inches='tight')


