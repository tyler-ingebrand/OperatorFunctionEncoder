import os

import numpy as np
import torch
from FunctionEncoder import *
# add src to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.MatrixMethodHelpers import get_hidden_layer_size
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"


device="cuda"
load_dir = 'logs_custom/Integral/matrix_least_squares/2024-11-06_12-02-40/'
raise Exception("""This script expects you have trained a B2B model with 3 basis functions for BOTH input and output function spaces. 
                By default, if you set n_basis=3, the input function space has 3 basis functions but 
                the output function space has 3+1 basis functions. This is purely for debugging reasons. 
                To use this script, training a B2B model with 3 basis functions in both input and output 
                spaces first. """
                )

with torch.no_grad():

    # get hidden size
    hidden_size = get_hidden_layer_size(target_n_parameters=500_000,
                                        model_type="matrix",
                                        n_basis=3,
                                        n_layers=4,
                                        src_input_space=(1,),
                                        src_output_space=(1,),
                                        tgt_input_space=(1,),
                                        tgt_output_space=(1,),
                                        transformation_type="linear",
                                        n_sensors=1000,
                                        dataset_type="deterministic",
                                        )

    # create nns
    src_model = FunctionEncoder(input_size=(1,),
                                    output_size=(1,),
                                    data_type="deterministic",
                                    n_basis=3,
                                    method="least_squares",
                                    model_kwargs={"n_layers": 4, "hidden_size": hidden_size},
                                    ).to(device)

    tgt_model = FunctionEncoder(input_size=(1,),
                                output_size=(1,),
                                data_type="deterministic",
                                n_basis=3,
                                method="least_squares",
                                model_kwargs={"n_layers": 4, "hidden_size": hidden_size},
                                ).to(device)

    # load models
    src_model.load_state_dict(torch.load(f"{load_dir}/src_model.pth", weights_only=True))
    tgt_model.load_state_dict(torch.load(f"{load_dir}/tgt_model.pth", weights_only=True))


    # forward pass Basis functions
    src_xs = torch.linspace(-1, 1, 1000).unsqueeze(1).to(device)
    tgt_xs = torch.linspace(-10, 10, 1000).unsqueeze(1).to(device)
    src_ys = src_model.model(src_xs)[:, 0, :]
    tgt_ys = tgt_model.model(tgt_xs)[:, 0, :]

    # plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(3):
        # get each Basis function
        src_xs2 = src_xs.cpu().numpy()
        tgt_xs2 = tgt_xs.cpu().numpy()
        src_ys2 = src_ys[:, i].cpu().numpy()
        tgt_ys2 = tgt_ys[:, i].cpu().numpy()

        # plot
        axs[0].plot(src_xs.cpu(), src_ys2, label=f"Basis {i}", linewidth=3)
        axs[1].plot(tgt_xs.cpu(), tgt_ys2, label=f"Basis {i}", linewidth=3)

        # now plot the Line of Best Fit. The src space is a quadratic Line, the target space is a cubic Line.
        a,b,c = np.polyfit(src_xs2.squeeze(), src_ys2, deg=2)
        a2,b2,c2,d2 = np.polyfit(tgt_xs2.squeeze(), tgt_ys2, deg=3)
        d2 = 0

        # plot
        axs[0].plot(src_xs.cpu(), a*src_xs.cpu()**2 + b*src_xs.cpu() + c, linestyle="--", color="black")
        axs[1].plot(tgt_xs.cpu(), a2*tgt_xs.cpu()**3 + b2*tgt_xs.cpu()**2 + c2*tgt_xs.cpu() + d2, linestyle="--", color="black", label="Line(s) of Best Fit"if i == 0 else None)

    # titles
    axs[0].set_title("Source Basis Functions")
    axs[1].set_title("Target Basis Functions")

    # labels
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("g(x)")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("h(y)")

    # legend
    axs[0].legend()
    # reorganize labels so Line of Best Fit is last
    handles, labels = axs[1].get_legend_handles_labels()
    label_order = "Basis 0", "Basis 1", "Basis 2", "Line(s) of Best Fit"
    axs[0].legend([handles[labels.index(label)] for label in label_order], label_order, frameon=False)


    # save
    plt.tight_layout()
    plt.savefig(f"{load_dir}/Basis_functions.png")
    plt.clf()


