import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
from tqdm import trange
try:
    from plotting_specs import colors, labels, titles
except:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from plotting_specs import colors, labels, titles


plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"


from src.Datasets.OperatorDataset import OperatorDataset

from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import multivariate_normal


def generate_dataset(plot=False):
    def permeability(s):
        return 0.2 + s**2

    def sample_source_function(n_points):
        # sample a Gaussian process
        l=0.04
        sigma=1.0
        
        def source_function(x):
            K = np.exp(-0.5 * (x[:, np.newaxis] - x[np.newaxis, :])**2 / l**2)
            ys = multivariate_normal.rvs(mean=np.zeros_like(x), cov=K)
            return ys



        return source_function

    # Finite difference solver
    def solve_fd(n_points, source_function):
        x = np.linspace(0, 1, n_points)
        dx = x[1] - x[0]
        u = source_function(x)
        s = np.zeros(n_points)

        for _ in range(100):
            kappa = permeability(s)
            main_diag = (kappa[1:] + kappa[:-1])/dx**2
            upper_diag = -kappa[1:-1]/dx**2
            lower_diag = -kappa[1:-1]/dx**2
            
            A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], 
                    shape=(n_points-2, n_points-2))
            A = csc_matrix(A)
            s_interior = spsolve(A, u[1:-1])
            s[1:-1] = s_interior
        
        return x, s, u


    # Solve and compare both methods
    n_data = 40
    n_functions = 1000
    np.random.seed(0)

    if plot:
        fig, axs = plt.subplots(4, 6, figsize=(15, 10), dpi=300)
        for i in range(24):
            source_function = sample_source_function(n_data)
            x_fd, s_fd, u_fd = solve_fd(n_data, source_function)
            # print(x_fd.shape, x_fem.shape)

            ax = axs[i//6, i%6]
            ax.plot(x_fd, s_fd, 'b', linewidth=3, label='FD Solution')
            ax.plot(x_fd, u_fd, 'g', linewidth=3, label='Source Term')
        os.makedirs('src/Datasets/Darcy', exist_ok=True)
        plt.legend()
        plt.savefig('src/Datasets/Darcy/visualize.pdf', format='pdf', dpi=300, bbox_inches='tight')
    else:
        xs, f_xs, ys, tf_ys = [], [], [], []
        for i in trange(n_functions, desc="Generating 1D Darcy dataset"):
            # generate darcy solution
            source_function = sample_source_function(n_data)
            x_fd, s_fd, u_fd = solve_fd(n_data, source_function)

            # pull out the operator terms
            x = x_fd
            f_x = u_fd
            y = x_fd
            tf_y = s_fd 

            # save
            xs.append(x)
            f_xs.append(f_x)
            ys.append(y)
            tf_ys.append(tf_y)

        # convert to tensor
        xs = torch.tensor(np.array(xs))
        f_xs = torch.tensor(np.array(f_xs))
        ys = torch.tensor(np.array(ys))
        tf_ys = torch.tensor(np.array(tf_ys))

        # train/test split
        n_train = int(0.8 * n_functions)
        train_xs, train_f_xs, train_ys, train_tf_ys = xs[:n_train], f_xs[:n_train], ys[:n_train], tf_ys[:n_train]
        test_xs, test_f_xs, test_ys, test_tf_ys = xs[n_train:], f_xs[n_train:], ys[n_train:], tf_ys[n_train:]

        # save
        train = {"x": train_xs, "f": train_f_xs, "y": train_ys, "tf_y": train_tf_ys}
        test = {"x": test_xs, "f": test_f_xs, "y": test_ys, "tf_y": test_tf_ys}
        torch.save(train, 'src/Datasets/Darcy/nonlineardarcy_train.pt')
        torch.save(test, 'src/Datasets/Darcy/nonlineardarcy_test.pt')




class DarcySrcDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        if test:
            mat = torch.load('./src/Datasets/Darcy/nonlineardarcy_test.pt', weights_only=True)
        else:
            mat = torch.load('./src/Datasets/Darcy/nonlineardarcy_train.pt', weights_only=True)


        # format xs and ys
        xs = mat["x"].unsqueeze(-1)
        ys = mat["f"].unsqueeze(-1)        

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
        mean, std = 0.0205183576, 0.3604165445
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
            mat = torch.load('./src/Datasets/Darcy/nonlineardarcy_test.pt', weights_only=True)
        else:
            mat = torch.load('./src/Datasets/Darcy/nonlineardarcy_train.pt', weights_only=True)


        # format xs and ys
        xs = mat["y"].unsqueeze(-1)
        ys = mat["tf_y"].unsqueeze(-1)

        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                         *args, **kwargs,
                         )

        # normalization, hard-coded constants for consistency
        mean, std = 0.0341198272, 1.0113011797
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

if __name__ == "__main__":
    generate_dataset(plot=False)