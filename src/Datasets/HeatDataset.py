import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch

from src.Datasets.OperatorDataset import OperatorDataset


# mat = scipy.io.loadmat('./src/Datasets/Heat/heatequation_train.mat')
# print(mat.keys())
# x = mat['x']
# y = mat['y']
# t = mat['t']
# u_train = mat['u_train']
# f_train = mat['f_train']
# alpha_train = mat['alpha_train']
# pass

class HeatSrcDataset(OperatorDataset):
    def __init__(self, test=False, *args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/Heat/heatequation_test.mat')
            temperature_tag = "f_test"
            alpha_tag = "alpha_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/Heat/heatequation_train.mat')
            temperature_tag = "f_train"
            alpha_tag = "alpha_train"

        # get input function space, which is actually just the constants of initial temperature and thermal diffusivity
        alpha = mat[alpha_tag]
        temperature = mat[temperature_tag]

        # now do some formatting
        xs = np.zeros_like(alpha).transpose()[:, None, :] # this is just a constant function with one point at 0
        ys = np.concatenate([alpha, temperature], axis=0).transpose()[:, None, :] # the output is the initial temperature and thermal diffusivity

        if "n_examples_per_sample" in kwargs and kwargs["n_examples_per_sample"] != ys.shape[1]:
            print(f"WARNING: n_examples_per_sample is hard set to {ys.shape[1]} for the Darcy Dataset.")
        kwargs["n_examples_per_sample"] = ys.shape[1]

        super().__init__(input_size=(1,),
                         output_size=(2,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                         *args, **kwargs,
                         )

        # normalization, hard-coded constants for consistency
        self.xs = torch.tensor(xs).to(torch.float32)
        self.ys = torch.tensor(ys).to(torch.float32)

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,))
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

class HeatTgtDataset(OperatorDataset):
    def __init__(self, test=False, *args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/Heat/heatequation_test.mat')
            u_tag = "u_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/Heat/heatequation_train.mat')
            u_tag = "u_train"

        # get the input space, x pos, y pos, and time
        x = mat["x"]
        y = mat["y"]
        t = mat["t"]

        # remove the first and last rows/columns, as they are boundary conditions and hard set to 0 always
        # there is no point in learning this
        x = x[:, 1:-1, 1:-1]
        y = y[:, 1:-1, 1:-1]
        t = t[:, 1:-1, 1:-1]

        # get the output space, which is the temperature at each point at each time
        u = mat[u_tag]
        u = u[:, :, 1:-1, 1:-1]

        # now do some formatting
        xs = np.stack([x, y, t], axis=-1)
        ys = u[:, :, :, :, None]

        # squeeze dims
        xs = xs.reshape(-1, 3)[None, :, :].repeat(250, 0) # .repeat(ys.shape[0], 1)
        ys = ys.reshape(ys.shape[0], -1, 1)

        kwargs['n_examples_per_sample'] = 1_000
        super().__init__(input_size=(3,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=10_000,
                         *args, **kwargs,
                         )

        # normalization, hard-coded constants for consistency
        # normally you would subtract mean here, but the sign means something, so we dont want to change the positive/negative aspect.
        ys = ys / 0.2319188117980957 # TODO this is unneeded.


        self.xs = torch.tensor(xs).to(torch.float32)
        self.ys = torch.tensor(ys).to(torch.float32)

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,))
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        function_indicies = info["function_indicies"]
        xs = self.xs[function_indicies]

        # now sample a subset of the x,y,t pos/time locations
        self.sample_indicies = torch.randint(0, self.xs.shape[1], (n_samples,))
        xs = xs[:, self.sample_indicies]
        # those indicies are stored for the output function

        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        ys = self.ys[function_indicies]
        ys = ys[:, self.sample_indicies]
        return ys

def plot_source_heat(xs, ys, y_hats, info, logdir):
    pass # no need to plot this, as it is just a constant function

def plot_target_heat(xs, ys, y_hats, info, logdir):
    width, height = 99, 99

    for env in range(xs.shape[0]):
        fig, axs = plt.subplots(2, 4, figsize=(10, 10))
        vmin = 0
        vmax = ys[env].max().item()
        for time in range(4):
            # get images
            xs_temp = xs[env, width*height*time:width*height*(time+1)]
            ys_temp = ys[env, width*height*time:width*height*(time+1)]
            y_hats_temp = y_hats[env, width*height*time:width*height*(time+1)]

            # make plots
            ax = axs[0, time]
            ax.imshow(ys_temp.cpu().reshape(width, height), vmin=vmin, vmax=vmax)
            ax.axis("off")
            ax.set_title("Groundtruth")

            ax = axs[1, time]
            ax.imshow(y_hats_temp.cpu().reshape(width, height), vmin=vmin, vmax=vmax)
            ax.axis("off")
            ax.set_title("Estimate")

        plt.tight_layout()
        plt.savefig(f"{logdir}/heat_{env}.png")


def plot_transformation_heat(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    width, height = 99, 99

    for env in range(xs.shape[0]):
        fig, axs = plt.subplots(2, 4, figsize=(10, 10))
        vmin = 0
        vmax = ys[env].max().item()
        for time in range(4):
            alpha, temperature = example_ys[env, 0, 0], example_ys[env, 0, 1]

            # get images
            xs_temp = xs[env, width * height * time:width * height * (time + 1)]
            ys_temp = ys[env, width * height * time:width * height * (time + 1)]
            y_hats_temp = y_hats[env, width * height * time:width * height * (time + 1)]

            # make plots
            ax = axs[0, time]
            ax.imshow(ys_temp.cpu().reshape(width, height), vmin=vmin, vmax=vmax)
            ax.axis("off")
            ax.set_title('$\\alpha: {:.2f}, T: {:.2f}$'.format(alpha, temperature))

            ax = axs[1, time]
            ax.imshow(y_hats_temp.cpu().reshape(width, height), vmin=vmin, vmax=vmax)
            ax.axis("off")
            ax.set_title("Estimate")

        plt.tight_layout()
        plt.savefig(f"{logdir}/transformed_heat_{env}.png")