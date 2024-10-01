from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
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

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--n_sensors", type=int, default=1000)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--approximate_number_paramaters", type=int, default=500_000)
parser.add_argument("--unfreeze_sensors", action="store_true")
parser.add_argument("--dataset_type", type=str, default="Derivative")

args = parser.parse_args()


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
transformation_type = "linear"
n_layers = args.n_layers
freeze_example_xs = not args.unfreeze_sensors
exp_dir = "logs_experiment"

# get load dir
base_matrix = os.path.join(exp_dir, dataset_type, "matrix_least_squares")
base_deeponet = os.path.join(exp_dir, dataset_type, "deeponet")

# get all dirts
all_deeponet_dirs = os.listdir(base_deeponet)
all_matrix_dirs = os.listdir(base_matrix)

# remove files
all_deeponet_dirs = [d for d in all_deeponet_dirs if os.path.isdir(os.path.join(base_deeponet, d))]
all_matrix_dirs = [d for d in all_matrix_dirs if os.path.isdir(os.path.join(base_matrix, d))]

# sort them by date
all_deeponet_dirs = sorted(all_deeponet_dirs, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))
all_matrix_dirs = sorted(all_matrix_dirs, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))

# errors to keep track of
b2b_ood_function_errors = []
b2b_linearity_function_errors = []
b2b_homogeneity_function_errors = []
deeponet_ood_function_errors = []
deeponet_linearity_function_errors = []
deeponet_homogeneity_function_errors = []

for i, (deeponet_dir, matrix_dir) in enumerate(zip(all_deeponet_dirs, all_matrix_dirs)):
    print(f"Running experiment {i+1}/{len(all_deeponet_dirs)}")
    
    # get the load paths
    load_path_matrix = os.path.join(base_matrix, matrix_dir)
    load_path_deeponet = os.path.join(base_deeponet, deeponet_dir)

    # seed
    torch.manual_seed(i+1) # this is based on the order of training. Must be correct for deeponet to work. 

    # first load datasets
    # we are doing derivative dataset for this linearity test
    kwargs = {"a_range": (-100, 100), "b_range": (-100, 100), "c_range": (-100, 100)}
    if dataset_type == "Derivative":
         kwargs["d_range"] = (-100, 100)

    _, _, testing_combined_dataset = get_dataset(dataset_type, test=True, model_type="matrix", n_sensors=args.n_sensors, device=device,   freeze_example_xs=freeze_example_xs, **kwargs)
    testing_combined_dataset.n_functions_per_sample = 100
    testing_combined_dataset.src_dataset.n_functions_per_sample = 100
    testing_combined_dataset.tgt_dataset.n_functions_per_sample = 100

    # create the matrix method
    hidden_size = get_hidden_layer_size(target_n_parameters=args.approximate_number_paramaters,
                                        model_type="matrix",
                                        n_basis=n_basis, n_layers=n_layers,
                                        src_input_space=testing_combined_dataset.src_dataset.input_size,
                                        src_output_space=testing_combined_dataset.src_dataset.output_size,
                                        tgt_input_space=testing_combined_dataset.tgt_dataset.input_size,
                                        tgt_output_space=testing_combined_dataset.tgt_dataset.output_size,
                                        transformation_type=transformation_type,
                                        n_sensors=testing_combined_dataset.n_examples_per_sample,
                                        dataset_type=dataset_type,)
    src_model = FunctionEncoder(input_size=testing_combined_dataset.src_dataset.input_size,
                                output_size=testing_combined_dataset.src_dataset.output_size,
                                data_type=testing_combined_dataset.src_dataset.data_type,
                                n_basis=n_basis,
                                method=args.train_method,
                                # regularization_parameter=100.0, # NOTE: this is necesarry because some of the datasets are un-normalized. The scale of the data can be large (e.g. 100k), and so the corresponding MSE is extremely large. This loss term than overrides the regularization. Typically, default regularization is sufficient though.
                                model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                ).to(device)
    tgt_model = FunctionEncoder(input_size=testing_combined_dataset.tgt_dataset.input_size,
                                output_size=testing_combined_dataset.tgt_dataset.output_size,
                                data_type=testing_combined_dataset.tgt_dataset.data_type,
                                n_basis=n_basis+1, # note this makes debugging way easier.
                                method=args.train_method,
                                # regularization_parameter=100.0,
                                model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                                ).to(device)
    matrix_model = {"src": src_model, "tgt": tgt_model}
    matrix_model["A"] = torch.rand(tgt_model.n_basis, src_model.n_basis).to(device)

    # load the matrix method
    matrix_model["src"].load_state_dict(torch.load(f"{load_path_matrix}/src_model.pth", weights_only=True))
    matrix_model["tgt"].load_state_dict(torch.load(f"{load_path_matrix}/tgt_model.pth", weights_only=True))
    matrix_model["A"] = torch.load(f"{load_path_matrix}/A.pth", weights_only=True).to(device)

    # create deeponet
    hidden_size = get_hidden_layer_size(target_n_parameters=args.approximate_number_paramaters,
                                        model_type="deeponet",
                                        n_basis=n_basis, n_layers=n_layers,
                                        src_input_space=testing_combined_dataset.src_dataset.input_size,
                                        src_output_space=testing_combined_dataset.src_dataset.output_size,
                                        tgt_input_space=testing_combined_dataset.tgt_dataset.input_size,
                                        tgt_output_space=testing_combined_dataset.tgt_dataset.output_size,
                                        transformation_type=transformation_type,
                                        n_sensors=testing_combined_dataset.n_examples_per_sample,
                                        dataset_type=dataset_type,)
    deeponet_model = DeepONet(input_size_tgt=testing_combined_dataset.tgt_dataset.input_size[0],
                        output_size_tgt=testing_combined_dataset.tgt_dataset.output_size[0],
                        input_size_src=testing_combined_dataset.src_dataset.input_size[0],
                        output_size_src=testing_combined_dataset.src_dataset.output_size[0],
                        n_input_sensors=testing_combined_dataset.n_examples_per_sample,
                        p=n_basis,
                        n_layers=n_layers,
                        hidden_size=hidden_size,
                        ).to(device)

    # load the deeponet
    deeponet_model.load_state_dict(torch.load(f"{load_path_deeponet}/model.pth", weights_only=True))

    with torch.no_grad():
        # now we can sample a function who is large in magnitude, then plot it. We can then compare the matrix method and the deeponet method.
        example_xs, example_ys, xs, ys, info = testing_combined_dataset.sample(device=None)
        old_example_xs = example_xs

        # first do matrix method
        # get its source fit
        matrix_method_example_y_hats = matrix_model["src"].predict_from_examples(example_xs, example_ys, example_xs, method=args.train_method)

        # get the target fit
        rep, _ = matrix_model["src"].compute_representation(example_xs, example_ys, method=args.train_method)
        rep = rep @ matrix_model["A"].T
        matrix_method_y_hats = matrix_model["tgt"].predict(xs, rep)

        # now do deeponet
        deeponet_y_hats = deeponet_model.forward(example_xs, example_ys, xs)

        # print the errors
        matrix_method_error = (matrix_method_y_hats - ys)**2
        deeponet_error = (deeponet_y_hats - ys)**2
        
        # append errors
        b2b_ood_function_errors.append(matrix_method_error.detach().cpu())
        deeponet_ood_function_errors.append(deeponet_error.detach().cpu())

        # change device of everything
        example_xs = example_xs.cpu()
        example_ys = example_ys.cpu()
        xs = xs.cpu()
        ys = ys.cpu()
        matrix_method_example_y_hats = matrix_method_example_y_hats.cpu()
        matrix_method_y_hats = matrix_method_y_hats.cpu()
        deeponet_y_hats = deeponet_y_hats.cpu()

        # sort
        sort_idx = xs.argsort(dim=1)
        xs = xs.gather(1, sort_idx)
        ys = ys.gather(1, sort_idx)
        matrix_method_y_hats = matrix_method_y_hats.gather(1, sort_idx)
        deeponet_y_hats = deeponet_y_hats.gather(1, sort_idx)

        sort_idx = example_xs.argsort(dim=1)
        example_xs = example_xs.gather(1, sort_idx)
        example_ys = example_ys.gather(1, sort_idx)
        matrix_method_example_y_hats = matrix_method_example_y_hats.gather(1, sort_idx)

        # plot the results
        b2b_color = colors["matrix_least_squares"]
        b2b_label = labels["matrix_least_squares"]
        deeponet_color = colors["deeponet"]
        deeponet_label = labels["deeponet"]

        # print the index of the worst case row
        errs = (matrix_method_y_hats - ys)**2
        errs = errs.mean(dim=(1,2))
        print(f"Worst case row: {errs.argmax().item()}")

        # two panes, one for the source, one for the target
        for row in (list(range(10)) + [errs.argmax().item()]):
            size=5
            fig = plt.figure(figsize=(4.25 * size, 1. * size), dpi=300)
            gridspec = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 1, 1])
            axs = gridspec.subplots()

            # plot source space
            ax = axs[0]
            ax.plot(example_xs[row].cpu(), example_ys[row].cpu(), label="Groundtruth", color="black")
            ax.plot(example_xs[row].cpu(), matrix_method_example_y_hats[row].cpu(), label=b2b_label, color=b2b_color)
            
            # title depending on data
            if dataset_type == "Derivative":
                title = f"${info['As'][row].item():.2f}x^3 + {info['Bs'][row].item():.2f}x^2 + {info['Cs'][row].item():.2f}x + {info['Ds'][row].item():.2f}$"
            elif dataset_type == "Integral":
                title = f"${info['As'][row].item():.2f}x^2 + {info['Bs'][row].item():.2f}x + {info['Cs'][row].item():.2f}$"
            elif dataset_type == "Darcy":
                title = f"$f(x) \\sim \\mathcal{{G}} \\mathcal{{P}}$"

            ax.set_title(title)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")

            # plot source error
            ax = axs[1]
            error = (matrix_method_example_y_hats[row] - example_ys[row]).abs()
            ax.plot(example_xs[row].cpu(), error.cpu(), label=b2b_label, color=b2b_color)
            ax.set_xlabel("$x$")
            ax.set_ylabel(f"$\\vert \hat{{f}}(x) - f(x) \\vert$")
            ax.set_yscale("log")
            title = f"Absolute Error"
            ax.set_title(title)



            # disable axs[2] so its just white
            ax = axs[2]
            ax.axis("off")


            # plot
            ax = axs[3]
            ax.plot(xs[row].cpu(), ys[row].cpu(), label="Groundtruth", color="black")
            ax.plot(xs[row].cpu(), matrix_method_y_hats[row].cpu(), label=b2b_label, color=b2b_color)
            ax.plot(xs[row].cpu(), deeponet_y_hats[row].cpu(), label=deeponet_label, color=deeponet_color, ls="--")
            
            if dataset_type == "Derivative":
                title = f"$3*{info['As'][row].item():.2f}x^2 + 2*{info['Bs'][row].item():.2f}x + {info['Cs'][row].item():.2f}$"
            elif dataset_type == "Integral":
                a_string = f"{info['As'][row].item():.2f}"
                b_string = f"{info['Bs'][row].item():.2f}"
                c_string = f"{info['Cs'][row].item():.2f}"
                title = f"$\\frac{{{a_string}}}{{3}}x^3 + \\frac{{{b_string}}}{{2}}x^2  + \\frac{{{c_string}}}{{1}}x$"        
            elif dataset_type == "Darcy":
                title = f"Darcy Solution"
            ax.set_title(title)


            leg = ax.legend(frameon=False)
            # change deeponet linestyle in the legend to be normal, even though its dashed
            for line in leg.get_lines():
                line.set_linestyle('-')

            ax.set_xlabel("$y$")
            ax.set_ylabel("$(\\mathcal{{T}}f)(y)$")


            # plot absolute error
            ax = axs[4]
            error = (matrix_method_y_hats[row] - ys[row]).abs()
            ax.plot(xs[row].cpu(), error.cpu(), label=b2b_label, color=b2b_color)
            error = (deeponet_y_hats[row] - ys[row]).abs()
            ax.plot(xs[row].cpu(), error.cpu(), label=deeponet_label, color=deeponet_color)
            ax.set_xlabel("$y$")
            ax.set_ylabel(f"$\\vert \hat{{\\mathcal{{T}}}}f(y) - \\mathcal{{T}}f(y) \\vert$")
            ax.set_yscale("log")
            title = f"Absolute Error"
            ax.set_title(title)

            # add line between ax2 and ax3
            left = axs[1].get_position().xmax 
            right = axs[3].get_position().xmin - 0.015
            xpos = (left+right) / 2
            top = axs[1].get_position().ymax + 0.08
            bottom = axs[1].get_position().ymin - 0.08
            line1 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top),transform=fig.transFigure, color="black", linestyle="--", lw=2)
            fig.lines = line1, 


            plt.tight_layout()
            plot_name = f"{load_path_matrix}/linearity_test_{row}.pdf"
            plt.savefig(plot_name)

            # close figs
            plt.close("all")




        # next, we are going to emprically validate that the representation computed is linear and homogenous. 
        n_functions = 100
        total = 10000
        n_samples = 1000

        # get the normal dataset
        torch.manual_seed(i+1) # this is based on the order of training. Must be correct for deeponet to work. 
        _, _, testing_combined_dataset = get_dataset(dataset_type, test=True, model_type="matrix", n_sensors=args.n_sensors, device=device,   freeze_example_xs=freeze_example_xs)
        
        # this is because we want to test a lot to compute an average
        testing_combined_dataset.n_functions_per_sample = n_functions
        testing_combined_dataset.src_dataset.n_functions_per_sample = n_functions
        testing_combined_dataset.tgt_dataset.n_functions_per_sample = n_functions

        for i in range(total//n_functions):
            # sample functions
            f1_info = testing_combined_dataset.src_dataset.sample_info()
            f2_info = testing_combined_dataset.src_dataset.sample_info()

            # sample scalars between -5 and 5
            r1, r2 = torch.rand((n_functions, 1), device=device)*10-5, torch.rand((n_functions, 1), device=device)*10-5

            # compute f3 = r1*f1 + r2*f2
            f3_info = {"As": r1*f1_info["As"] + r2*f2_info["As"],
                        "Bs": r1*f1_info["Bs"] + r2*f2_info["Bs"],
                        "Cs": r1*f1_info["Cs"] + r2*f2_info["Cs"],
                        }
            if dataset_type == "Derivative":
                f3_info["Ds"] = r1*f1_info["Ds"] + r2*f2_info["Ds"]
            
            # compute f4 = r1 * f1
            f4_info = {"As": r1*f1_info["As"],
                        "Bs": r1*f1_info["Bs"],
                        "Cs": r1*f1_info["Cs"],
                        }
            if dataset_type == "Derivative":
                f4_info["Ds"] = r1*f1_info["Ds"]
            
            # sample example xs
            example_xs, _, _, _, _ = testing_combined_dataset.sample(device=None)
            assert torch.all(example_xs == old_example_xs)

            # compute outputs for all datasets
            f1_example_ys = testing_combined_dataset.src_dataset.compute_outputs(f1_info, example_xs)
            f2_example_ys = testing_combined_dataset.src_dataset.compute_outputs(f2_info, example_xs)

            # compute representations
            f1_rep, _ = matrix_model["src"].compute_representation(example_xs, f1_example_ys, method=args.train_method)
            f2_rep, _ = matrix_model["src"].compute_representation(example_xs, f2_example_ys, method=args.train_method)
            deeponet_f1_rep = deeponet_model.forward_branch(f1_example_ys)
            deeponet_f2_rep = deeponet_model.forward_branch(f2_example_ys)

            # move to target space.
            f1_rep = f1_rep @ matrix_model["A"].T
            f2_rep = f2_rep @ matrix_model["A"].T

            # compute the linear combination
            f3_rep_hat = r1*f1_rep + r2*f2_rep
            f4_rep_hat = r1*f1_rep
            deeponet_f3_rep_hat = r1*deeponet_f1_rep + r2*deeponet_f2_rep
            deeponet_f4_rep_hat = r1*deeponet_f1_rep


            # groundtruth Tf
            xs = testing_combined_dataset.tgt_dataset.sample_inputs(f3_info, 10_000)
            f3 = testing_combined_dataset.tgt_dataset.compute_outputs(f3_info, xs)
            f4 = testing_combined_dataset.tgt_dataset.compute_outputs(f4_info, xs)

            # approximate Tf
            f3_hat = matrix_model["tgt"].predict(xs, f3_rep_hat)
            f4_hat = matrix_model["tgt"].predict(xs, f4_rep_hat)

            # def forward(self, xs, us, ys): # deeponet forward function, we only need half of it here. 
                # # xs are not actually used for deeponet, but we keep them to be consistent with the function encoder
                # # us are the values of u at the input sensors
                # # ys are the locations of the output sensors.
                # b = self.forward_branch(us)
                # t = self.forward_trunk(ys)

                # # this is just the dot product, but allowing for the output dim to be > 1
                # G_u_y = torch.einsum("fpz,fdpz->fdz", b, t)

                # # optionally add bias
                # if self.bias is not None:
                #     G_u_y = G_u_y + self.bias

                # return G_u_y
            t = deeponet_model.forward_trunk(xs)
            G_u_y = torch.einsum("fpz,fdpz->fdz", deeponet_f3_rep_hat, t)
            if deeponet_model.bias is not None:
                G_u_y = G_u_y + deeponet_model.bias
            deeponet_f3_hat = G_u_y

            t = deeponet_model.forward_trunk(xs)
            G_u_y = torch.einsum("fpz,fdpz->fdz", deeponet_f4_rep_hat, t)
            if deeponet_model.bias is not None:
                G_u_y = G_u_y + deeponet_model.bias
            deeponet_f4_hat = G_u_y

            
            # now log errors

            # compute matrix method f3 error
            dif = f3_hat - f3
            b2b_linearity_function_errors.append((dif**2).cpu())

            # compute deeponet f3 error
            dif = deeponet_f3_hat - f3
            deeponet_linearity_function_errors.append((dif**2).cpu())

            # compute matrix method f4 error
            dif = f4_hat - f4
            b2b_homogeneity_function_errors.append((dif**2).cpu())

            # compute deeponet f4 error
            dif = deeponet_f4_hat - f4
            deeponet_homogeneity_function_errors.append((dif**2).cpu())

           


# flatten all datasets:
b2b_ood_function_errors = torch.cat(b2b_ood_function_errors).flatten()
b2b_linearity_function_errors = torch.cat(b2b_linearity_function_errors).flatten()
b2b_homogeneity_function_errors = torch.cat(b2b_homogeneity_function_errors).flatten()
deeponet_ood_function_errors = torch.cat(deeponet_ood_function_errors).flatten()
deeponet_linearity_function_errors = torch.cat(deeponet_linearity_function_errors).flatten()
deeponet_homogeneity_function_errors = torch.cat(deeponet_homogeneity_function_errors).flatten()

# ${mean_str[0]}\\mathrm{{e}}{{{mean_str[1]}}} \\pm {std_str[0]}\\mathrm{{e}}{{{std_str[1]}}}$
def plot_(tag, mean, std):
    mean_str = f"{mean:0.2E}".split("E")
    std_str = f"{std:0.2E}".split("E")
    print(tag, "\t", end="")
    print(" & ", end="")
    print(f"${mean_str[0]}\\mathrm{{e}}{{{mean_str[1]}}} \\pm {std_str[0]}\\mathrm{{e}}{{{std_str[1]}}}$", end="")
    print(" & ", end="\n")


# print the errors
print(f"{dataset_type}:")
plot_("B2B OOD Function Error                   ", torch.mean((b2b_ood_function_errors)).item(), torch.std((b2b_ood_function_errors)).item())
plot_("DeepONet OOD Function Error              ", torch.mean((deeponet_ood_function_errors)).item(), torch.std((deeponet_ood_function_errors)).item())
plot_("B2B Linearity Function Error             ", torch.mean((b2b_linearity_function_errors)).item(), torch.std((b2b_linearity_function_errors)).item())
plot_("DeepONet Linearity Function Error        ", torch.mean((deeponet_linearity_function_errors)).item(), torch.std((deeponet_linearity_function_errors)).item())
plot_("B2B Homogeneity Function Error           ", torch.mean((b2b_homogeneity_function_errors)).item(), torch.std((b2b_homogeneity_function_errors)).item())
plot_("DeepONet Homogeneity Function Error      ", torch.mean((deeponet_homogeneity_function_errors)).item(), torch.std((deeponet_homogeneity_function_errors)).item())
print("\n")
