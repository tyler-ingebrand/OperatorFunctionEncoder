




import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.SVDEncoder import SVDEncoder
import matplotlib.pyplot as plt
from plotting_specs import colors, labels, titles
from scipy.optimize import curve_fit
import numpy as np


experiment_dir = "logs_experiment"


for dataset in ["Integral", "Derivative"]:
    for alg in ["matrix_least_squares", "Eigen_least_squares", "SVD_least_squares"]:
        load_dir = f"{experiment_dir}/{dataset}/{alg}"

        # we are going to load all the eigen exampes, and print the calculated eigen values. 
        all_sigma_values = []
        for subdir in os.listdir(load_dir):
            if not os.path.isdir(f"{load_dir}/{subdir}"):
                continue
            
            if alg == "Eigen_least_squares" or alg == "SVD_least_squares":
                # load the hyper params to generate the model
                params = torch.load(f"{load_dir}/{subdir}/params.pth", weights_only=True)
                n_layers, hidden_size, n_basis = params["n_layers"], params["hidden_size"], params["n_basis"]
                
                
                # the model we will load everything into 
                model = SVDEncoder(input_size_src=(1,),
                                output_size_src=(1,),
                                input_size_tgt=(1,),
                                output_size_tgt=(1,),
                                data_type="deterministic", # we dont support stochastic for now, though its possible.
                                n_basis=n_basis,
                                method="least_squares",
                                use_eigen_decomp=False,
                                model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size})

                # load model
                model.load_state_dict(torch.load(f"{load_dir}/{subdir}/model.pth", weights_only=True))

                # fetch eigen values
                sigma_values = model.sigma_values.detach()

                # take abs since sign is arbitrary
                sigma_values = torch.abs(sigma_values)

                # sort them in descending order
                sigma_values = sigma_values.sort(descending=True)[0]
            elif alg == "matrix_least_squares":
                # load the hyper params to generate the model
                params = torch.load(f"{load_dir}/{subdir}/params.pth", weights_only=True)
                n_layers, hidden_size, n_basis = params["n_layers"], params["hidden_size"], params["n_basis"]
                
                
                # the model we will load everything into 
                A = torch.load(f"{load_dir}/{subdir}/A.pth", weights_only=True).to("cpu")

                # do a svd on A
                sigma = torch.svd(A, compute_uv=False).S

                # take abs since sign is arbitrary
                sigma_values = torch.abs(sigma)

                # sort them in descending order
                sigma_values = sigma_values.sort(descending=True)[0]
            else:
                raise ValueError("Invalid alg")

            # plot the decay of sigma values
            all_sigma_values.append(sigma_values)

        # compute median, third and first quartiles
        sigma_values = torch.stack(all_sigma_values)
        sigma_values = sigma_values
        median = torch.median(sigma_values, dim=0).values
        q1 = torch.quantile(sigma_values, 0.25, dim=0)
        q3 = torch.quantile(sigma_values, 0.75, dim=0)

        # fit an exponential curve to the medians
        x = torch.arange(start=0, end=len(median), dtype=torch.float32)
        y = median
        logy = torch.log(y)
        # now we have linear regression between x, logy
        # compute the exact solution
        constants = torch.ones_like(x)
        A = torch.stack([x, constants], dim=1)
        b = logy
        # solve the system
        solution = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.flatten()
        a, b = solution[1], solution[0]

        # send to cpu
        a, b = a.cpu(), b.cpu()

        # def a model for plotting
        def model(x, a, b, c):
            return torch.exp(b*x + a)

        # plot the decay of sigma values
        plt.figure()
        plt.plot(median, label="Median")
        plt.fill_between(range(len(median)), q1, q3, alpha=0.5)

        # plot the fitted exponential curve
        a_str, b_str = f"{torch.exp(a).item():.3f}", f"{b.item():.3f}"
        label = r"$" + a_str + r"e^{" + b_str + r"x}$"
        print(label)
        # label = "Line of Best Fit"
        plt.plot(x, model(x, a, b, 0), label=label, ls="--", color="black")

        # labels
        plt.legend(frameon=False)
        plt.xlabel("Index")
        ylabel = "Absolute Eigen Value" if alg == "Eigen_least_squares" else "Absolute Singular Value"
        plt.ylabel(ylabel)
        title = f"{'Matrix Singular' if alg == 'matrix_least_squares' else 'Singular' if alg == 'SVD_least_squares' else 'Eigen'} Value Decay on the {titles[dataset]} Dataset"
        plt.title(title)

        plt.savefig(f"{load_dir}/{dataset}_{alg}_sigma_decay.pdf")