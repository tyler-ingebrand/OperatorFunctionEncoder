import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
from plotting_specs import colors, labels, titles

from src.Datasets.OperatorDataset import OperatorDataset
plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

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
    def __init__(self, test=False, device="cuda", *args, **kwargs):
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
            print(f"WARNING: n_examples_per_sample is hard set to {ys.shape[1]} for the Heat Dataset.")
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
        self.xs = torch.tensor(xs).to(torch.float32).to(device)
        self.ys = torch.tensor(ys).to(torch.float32).to(device)
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

class HeatTgtDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
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
        xs = xs.reshape(-1, 3)[None, :, :].repeat(ys.shape[0], 0) # .repeat(ys.shape[0], 1)
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
        # dont want to shift by mean because the sign means something.
        std = 0.2342301309108734
        ys = ys / std

        self.xs = torch.tensor(xs).to(torch.float32).to(device)
        self.ys = torch.tensor(ys).to(torch.float32).to(device)
        self.sample_indicies = None
        self.device = device

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,), device=self.device)
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        function_indicies = info["function_indicies"]
        xs = self.xs[function_indicies]

        # now sample a subset of the x,y,t pos/time locations
        if not self.freeze_xs:
            self.sample_indicies = torch.randint(0, self.xs.shape[1], (n_samples,), device=self.device)
        elif self.sample_indicies is None:
            self.sample_indicies = torch.randint(0, self.xs.shape[1], (n_samples,), device=self.device)

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
    model_type = info["model_type"]
    color = colors[model_type]
    label = labels[model_type]
    size=5

    # undo normalization for visualization purposes
    std = 0.2342301309108734
    ys = ys * std
    y_hats = y_hats * std

    for env in range(xs.shape[0]):
        fig = plt.figure(figsize=(2.8 * size, 1 * size), dpi=300)
        # gridspec = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.12, 1, 0.12])
        # axs = gridspec.subplots()
        
        # create 3 types of plots
        gridspec_left = fig.add_gridspec(1, 2, )
        gridspec_cb1 = fig.add_gridspec(1, 1, )
        gridspec_right = fig.add_gridspec(1, 1, )
        gridspec_cb2 = fig.add_gridspec(1, 1, )
        
        # compute boundaries. 
        width_ratios=[1, 1, 0.11, 1, 0.11]
        start = 0.05
        stop = 0.93
        wspace = 0.01
        available_space = stop - start - wspace * (len(width_ratios) - 1)
        width = available_space / sum(width_ratios)

        left1 = start
        right1 = start + width * 2
        left2 = right1 + wspace * 0.00
        right2 = left2 + 0.11 * width
        left3 = right2 + 4.5 * wspace
        right3 = left3 + width
        left4 = right3 + wspace * 0.0
        right4 = left4 + 0.11 * width
        gridspec_left.update(left=left1, right=right1, wspace=0.000)
        gridspec_cb1.update(left=left2, right=right2)
        gridspec_right.update(left=left3, right=right3)
        gridspec_cb2.update(left=left4, right=right4)


        vmin = 0
        vmax = ys[env].max().item()
        alpha, temperature = example_ys[env, 0, 0], example_ys[env, 0, 1]
        
        # plot the heat with time on the x axis and x dim on the y axis
        # groundtruth
        ax = fig.add_subplot(gridspec_left[0])
        intensity = ys[env, :, 0]
        intensity = intensity.reshape(80, 99).cpu().T
        ax.imshow(intensity, vmin=vmin, vmax=vmax)
        ax.set_title('Groundtruth ($\\alpha={:.2f}, T={:.2f}$)'.format(alpha, temperature))
        ax.set_xlabel("Time")
        ax.set_ylabel("X Position")
        # set time to go from 0 to 1
        # and pos to go from 0 to 1
        ax.set_xticks([0, 20, 40, 60, 79])
        ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 33, 66, 98])
        ax.set_yticklabels([0, 0.33, 0.66, 1])

        # estimate
        ax = fig.add_subplot(gridspec_left[1])
        intensity = y_hats[env, :, 0]
        intensity = intensity.reshape(80, 99).cpu().T
        ax.imshow(intensity, vmin=vmin, vmax=vmax)
        ax.set_title("{label} Estimate".format(label=label))
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.set_xticks([0, 20, 40, 60, 79])
        ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1])

        # do a color bar
        cax = fig.add_subplot(gridspec_cb1[0])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)
        cax.set_ylabel("Temperature")


        # next plot the error
        ax = fig.add_subplot(gridspec_right[0])
        error = torch.abs(ys[env, :, 0] - y_hats[env, :, 0])
        error = error.reshape(80, 99).cpu().T
        vmin = 0
        vmax = (ys[env, :, 0].max().item() - ys[env, :, 0].min().item()) * 0.1
        ax.imshow(error, vmin=vmin, vmax=vmax)
        ax.set_title("Absolute Error")
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.set_xticks([0, 20, 40, 60, 79])
        ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
        
        # do a color bar
        cax = fig.add_subplot(gridspec_cb2[0])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)
        cax.set_ylabel("Absolute Error")
        
        # plt.tight_layout(w_pad=0.2)
        plot_name = f"{logdir}/qualitative_Heat_{label.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}_{env}.png"
        plt.savefig(plot_name)
        plt.clf()
