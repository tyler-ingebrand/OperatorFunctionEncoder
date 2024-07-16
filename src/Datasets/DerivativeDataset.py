from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from matplotlib import pyplot as plt

from src.Datasets.OperatorDataset import OperatorDataset


class CubicDataset(OperatorDataset):

    def __init__(self,
                 a_range=(-3, 3),
                 b_range=(-3, 3),
                 c_range=(-3, 3),
                 d_range=(-3, 3),
                 input_range=(-10, 10),
                 *args,
                 **kwargs
                 ):
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         *args, **kwargs)
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.d_range = d_range
        self.input_range = input_range

    # the info dict is used to generate data. So first we generate an info dict
    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        As = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        Bs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        Cs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        Ds = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.d_range[1] - self.d_range[0]) + self.d_range[0]
        return {"As": As, "Bs": Bs, "Cs": Cs, "Ds": Ds}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        xs = torch.rand((info["As"].shape[0], n_samples, *self.input_size), dtype=torch.float32)
        xs = xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        # returns the cubic function ax^3 + bx^2 + cx + d
        As, Bs, Cs = info["As"], info["Bs"], info["Cs"]
        ys = As.unsqueeze(1) * inputs ** 3 + Bs.unsqueeze(1) * inputs ** 2 + Cs.unsqueeze(1) * inputs + info["Ds"].unsqueeze(1)
        return ys


class CubicDerivativeDataset(OperatorDataset):

    def __init__(self,
                 a_range=(-3, 3),
                 b_range=(-3, 3),
                 c_range=(-3, 3),
                 d_range=(-3, 3),
                 input_range=(-10, 10),
                 *args,
                 **kwargs
                 ):
        super().__init__(input_size=(1,), output_size=(1,), *args, **kwargs)
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.d_range = d_range
        self.input_range = input_range

    # the info dict is used to generate data. So first we generate an info dict
    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        As = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        Bs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        Cs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        Ds = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.d_range[1] - self.d_range[0]) + self.d_range[0]
        return {"As": As, "Bs": Bs, "Cs": Cs, "Ds": Ds}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        xs = torch.rand((info["As"].shape[0], n_samples, *self.input_size), dtype=torch.float32)
        xs = xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        # returns the derivative of a cubic function ax^3 + bx^2 + cx + d
        As, Bs, Cs = info["As"], info["Bs"], info["Cs"]
        ys = 3 * As.unsqueeze(1) * inputs ** 2 + 2 * Bs.unsqueeze(1) * inputs + Cs.unsqueeze(1)
        return ys

def plot_source_cubic(xs, ys, y_hats, info, logdir):
    # sort xs,ys,y_hats
    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    y_hats = y_hats.gather(dim=-2, index=indicies)

    # plot
    n_plots = 9
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(xs[i].cpu(), ys[i].cpu(), label="Groundtruth")
        ax.plot(xs[i].cpu(), y_hats[i].cpu(), label="Estimate")
        title = f"${info['As'][i].item():.2f}x^3 + {info['Bs'][i].item():.2f}x^2 + {info['Cs'][i].item():.2f}x + {info['Ds'][i].item():.2f}$"
        ax.set_title(title)
        if i == 8:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{logdir}/source.png")
    plt.clf()

def plot_target_cubic_derivative(xs, ys, y_hats, info, logdir):
    # sort xs,ys,y_hats
    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    y_hats = y_hats.gather(dim=-2, index=indicies)

    # plot
    n_plots = 9
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(xs[i].cpu(), ys[i].cpu(), label="Groundtruth")
        ax.plot(xs[i].cpu(), y_hats[i].cpu(), label="Estimate")
        title = f"$3*{info['As'][i].item():.2f}x^2 + 2*{info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
        ax.set_title(title)
        if i == 8:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{logdir}/target.png")
    plt.clf()

def plot_transformation_derivative(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    size = 5
    fig = plt.figure(figsize=(2.2 * size, 2.5 * size), dpi=300)
    gridspec = fig.add_gridspec(4, 3, width_ratios=[1, 0.4, 1])
    axs = gridspec.subplots()

    # sort data based on xs
    example_xs, indicies = torch.sort(example_xs, dim=-2)
    example_ys = example_ys.gather(dim=-2, index=indicies)
    if example_y_hats is not None:
        example_y_hats = example_y_hats.gather(dim=-2, index=indicies)

    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    y_hats = y_hats.gather(dim=-2, index=indicies)


    for row in range(4):

        # plot
        ax = axs[row, 0]
        ax.plot(example_xs[row].cpu(), example_ys[row].cpu(), label="Groundtruth")
        if example_y_hats is not None:
            ax.plot(example_xs[row].cpu(), example_y_hats[row].cpu(), label="Estimate")
        title = f"${info['As'][row].item():.2f}x^3 + {info['Bs'][row].item():.2f}x^2 + {info['Cs'][row].item():.2f}x + {info['Ds'][row].item():.2f}$"
        ax.set_title(title)


        # add an arrow to the middle column
        # and a T right above it
        ax = axs[row, 1]
        ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        ax.text(0.1, 0.1, "T", fontsize=30)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.axis("off")

        # plot
        ax = axs[row, 2]
        ax.plot(xs[row].cpu(), ys[row].cpu(), label="Groundtruth")
        ax.plot(xs[row].cpu(), y_hats[row].cpu(), label="Estimate")
        title = f"$3*{info['As'][row].item():.2f}x^2 + 2*{info['Bs'][row].item():.2f}x + {info['Cs'][row].item():.2f}$"
        ax.set_title(title)
        if row == 3:
            ax.legend()
    plt.tight_layout()
    plt.savefig(f"{logdir}/transformation.png")