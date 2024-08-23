import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
from plotting_specs import colors, labels, titles
plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

from src.Datasets.OperatorDataset import OperatorDataset

class DarcySrcDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/Darcy/nonlineardarcy_test.mat')
            tag = "f_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/Darcy/nonlineardarcy_train.mat')
            tag = "f_train"


        # format xs and ys
        ys = torch.tensor(mat[tag][:, :, None])
        xs = torch.tensor(mat["x"]).float().reshape(1, -1, 1).repeat(ys.shape[0], 1, 1)

        if "n_examples_per_sample" in kwargs and kwargs["n_examples_per_sample"] != ys.shape[1]:
            print(f"WARNING: n_examples_per_sample is hard set to {ys.shape[1]} for the Darcy Dataset.")
        kwargs["n_examples_per_sample"] = ys.shape[1]

        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                         *args, **kwargs,
                         )

        # normalization, hard-coded constants for consistency
        mean, std = (-0.02268278824048518, 0.48610630605316585)
        ys = (ys - mean) / std
        self.xs = xs.to(torch.float32).to(device)
        self.ys = ys.to(torch.float32).to(device)
        self.device = device

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,), device=self.device)
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        function_indicies = info["function_indicies"]
        xs = self.xs[function_indicies]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        ys = self.ys[function_indicies]
        return ys

class DarcyTgtDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/Darcy/nonlineardarcy_test.mat')
            tag = "u_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/Darcy/nonlineardarcy_train.mat')
            tag = "u_train"

        # format xs and ys
        ys = torch.tensor(mat[tag][:, :, None])
        xs = torch.tensor(mat["x"]).float().reshape(1, -1, 1).repeat(ys.shape[0], 1, 1)

        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                         *args, **kwargs,
                         )

        # normalization, hard-coded constants for consistency
        mean, std = (-0.008404619637732641, 0.14963929882038432)
        ys = (ys - mean) / std

        self.xs = xs.to(torch.float32).to(device)
        self.ys = ys.to(torch.float32).to(device)
        self.device = device

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,), device=self.device)
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        function_indicies = info["function_indicies"]
        xs = self.xs[function_indicies]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        ys = self.ys[function_indicies]
        return ys

def plot_source_darcy(xs, ys, y_hats, info, logdir):
    # plot
    n_plots = 9
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(xs[i].cpu(), ys[i].cpu(), label="Groundtruth")
        ax.plot(xs[i].cpu(), y_hats[i].cpu(), label="Estimate")
        if i == 8:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{logdir}/source.png")
    plt.clf()

def plot_target_darcy(xs, ys, y_hats, info, logdir):
    # plot
    n_plots = 9
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(xs[i].cpu(), ys[i].cpu(), label="Groundtruth")
        ax.plot(xs[i].cpu(), y_hats[i].cpu(), label="Estimate")
        if i == 8:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{logdir}/target.png")
    plt.clf()


def plot_transformation_darcy(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    size = 5
    model_type = info["model_type"]
    color = colors[model_type]
    label = labels[model_type]


    for row in range(example_xs.shape[0]):
        # create plot
        fig = plt.figure(figsize=(2.2 * size, 1 * size), dpi=300)
        gridspec = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        axs = gridspec.subplots()
        
        
        # plot
        ax = axs[0]
        ax.plot(example_xs[row].cpu(), example_ys[row].cpu(), label="Groundtruth", color="black")
        if example_y_hats is not None:
            ax.plot(example_xs[row].cpu(), example_y_hats[row].cpu(), label=label, color=color)
        # ax.legend()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(x)$")
        ax.set_title("Source term")

        # add an arrow to the middle column
        # and a T right above it
        # ax = axs[row, 1]
        # ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        # ax.text(0.1, 0.1, "T", fontsize=30)
        # ax.set_xlim(0, 0.5)
        # ax.set_ylim(-0.3, 0.3)
        # ax.axis("off")

        # plot
        ax = axs[1]
        ax.plot(xs[row].cpu(), ys[row].cpu(), label="Groundtruth", color="black")
        ax.plot(xs[row].cpu(), y_hats[row].cpu(), label=label, color=color)
        ax.legend()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$s(x)$")
        ax.set_title("Solution")

        plt.tight_layout()
        plot_name = f"{logdir}/qualitative_Darcy_{label.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}_{row}.pdf"
        plt.savefig(plot_name)
        plt.clf()