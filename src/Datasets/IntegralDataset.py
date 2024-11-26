from typing import Tuple, Union
from plotting_specs import colors, labels, titles

import torch
from matplotlib import pyplot as plt
from src.Datasets.OperatorDataset import OperatorDataset
plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
from plotting_specs import colors, labels, titles

class QuadraticIntegralDataset(OperatorDataset):

    def __init__(self,
                 a_range=(-3/5, 3/5),
                 b_range=(-3/5, 3/5),
                 c_range=(-3/5, 3/5),
                 input_range=(-10, 10),
                 device="cuda",
                 *args,
                 **kwargs
                 ):
        super().__init__(input_size=(1,), output_size=(1,),*args, **kwargs)
        self.a_range = torch.tensor(a_range, dtype=torch.float32, device=device)
        self.b_range = torch.tensor(b_range, dtype=torch.float32, device=device)
        self.c_range = torch.tensor(c_range, dtype=torch.float32, device=device)
        self.input_range = torch.tensor(input_range, dtype=torch.float32, device=device)
        self.device = device


    # the info dict is used to generate data. So first we generate an info dict
    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        As = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32, device=self.device) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        Bs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32, device=self.device) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        Cs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32, device=self.device) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]

        return {"As": As, "Bs": Bs, "Cs": Cs}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        xs = torch.rand((info["As"].shape[0], n_samples, *self.input_size), dtype=torch.float32, device=self.device)
        xs = xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        # returns the integral of a quadratic function ax^2 + bx + c
        As, Bs, Cs = info["As"], info["Bs"], info["Cs"]
        ys = 1/3. * As.unsqueeze(1) * inputs ** 3 + 1/2. * Bs.unsqueeze(1) * inputs ** 2 + Cs.unsqueeze(1) * inputs
        return ys

def plot_target_quadratic_integral(xs, ys, y_hats, info, logdir):
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
        title = f"$\\frac{{{info['As'][i].item():.2f}x^3}}{{3}} + \\frac{{{info['Bs'][i].item():.2f}x^2}}{{2}}  + {info['Cs'][i].item():.2f}x$"
        ax.set_title(title)
        if i == 8:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{logdir}/target.png")
    plt.clf()

def plot_transformation_integral(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    size = 5
    # ploting info
    model_type = info["model_type"]
    color = colors[model_type]
    label = labels[model_type]

    # sort data based on xs
    example_xs, indicies = torch.sort(example_xs, dim=-2)
    example_ys = example_ys.gather(dim=-2, index=indicies)
    if example_y_hats is not None:
        example_y_hats = example_y_hats.gather(dim=-2, index=indicies)

    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    y_hats = y_hats.gather(dim=-2, index=indicies)

    for row in range(example_xs.shape[0]):
        fig = plt.figure(figsize=(2.2 * size, 1 * size), dpi=300)
        gridspec = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        axs = gridspec.subplots()

        # plot
        ax = axs[0]
        ax.plot(example_xs[row].cpu(), example_ys[row].cpu(), label="Groundtruth", color="black")
        if example_y_hats is not None:
            ax.plot(example_xs[row].cpu(), example_y_hats[row].cpu(), label=label, color=color)
        title = f"${info['As'][row].item():.2f}x^2 + {info['Bs'][row].item():.2f}x + {info['Cs'][row].item():.2f}$"
        ax.set_title(title)


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
        a_string = f"{info['As'][row].item():.2f}"
        b_string = f"{info['Bs'][row].item():.2f}"
        c_string = f"{info['Cs'][row].item():.2f}"
        title = f"$\\frac{{{a_string}}}{{3}}x^3 + \\frac{{{b_string}}}{{2}}x^2  + \\frac{{{c_string}}}{{1}}x$"
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        plot_name = f"{logdir}/qualitative_Integral_{label.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}_{row}.pdf"
        plt.savefig(plot_name)