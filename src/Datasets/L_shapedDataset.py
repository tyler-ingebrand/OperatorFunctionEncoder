import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
from scipy.interpolate import griddata
from plotting_specs import colors, labels, titles

from src.Datasets.OperatorDataset import OperatorDataset
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# mat = scipy.io.loadmat('./src/Datasets/L-Shaped/linearDarcy_test.mat')
# # src spaces
# x_inp = mat['x_inp']
# y_inp = mat['y_inp']
# f1_test = mat['f1_test']
# f2_test = mat['f2_test']
#
# # plot f1 and f2 as images
# img1 = f1_test[0]
# img2 = f2_test[0]
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(img1)
# axs[1].imshow(img2)
# plt.savefig("f1_f2.png")
#
# # tgt spaces
# x = mat['x']
# y = mat['y']
# u_test = mat['u_test']
#
# # plot u as image
# plt.clf()
# plt.scatter(x, y, c=u_test[0])
# plt.savefig("u.png")
# exit(-1)


class LSrcDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda",*args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/L-Shaped/linearDarcy_test.mat')
            tags = "f1_test" , "f2_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/L-Shaped/linearDarcy_train.mat')
            tags = "f1_train", "f2_train"

        # fetch data
        x_inp = mat['x_inp'] # 31x31
        y_inp = mat['y_inp'] # 31x31
        f1 = mat[tags[0]] # fx31x31
        f2 = mat[tags[1]] # fx31x31

        # Comment: This data has a patch that is all zeros, ie no data
        # Deeponet uses a CNN, and so needs this patch as zeros.
        # However, we dont want to train the FE on this patch, as it is not useful.
        # so the example xs, example ys includes this patch as zeros
        # the xs, ys does NOT include this data
        # this change only affects the matrix method, and is necessary because otherwise the function space has
        # a discontinuity, which cannot be learned by a NN.
        # note that by including a patch of all zeros in the example data, this also effectively scales the inner product
        # by a factor of ~3/4, as 1/4 of the sample mean is all 0's
        # this is mathematically fine, as IPs can be scaled arbitrarily.

        
        # First do inputs
        ins = torch.tensor(np.concatenate([x_inp[:, :, None], y_inp[:, :, None]], axis=2), dtype=torch.float32)
        self.perm_example_xs = ins.reshape(1, -1, 2).expand(f1.shape[0], -1, -1)
        self.xs = torch.concat((ins[:,:16, :].reshape(-1, 2),
                               ins[:16, 16:, :].reshape(-1, 2)), dim=0).reshape(1, -1, 2).expand(f1.shape[0], -1, -1)

        # Now do outputs
        outs = torch.tensor(np.concatenate([f1[:, :, :, None], f2[:, :, :, None]], axis=3), dtype=torch.float32)
        self.perm_example_ys = outs.reshape(outs.shape[0], -1, 2)
        self.ys = torch.cat((outs[:, :, :16, :].reshape(outs.shape[0], -1, 2),
                                outs[:, :16, 16:, :].reshape(outs.shape[0], -1, 2)), dim=1)



        if "n_examples_per_sample" in kwargs and kwargs["n_examples_per_sample"] != self.perm_example_xs.shape[1]:
            print(f"WARNING: n_examples_per_sample is hard set to {self.perm_example_xs.shape[1]} for the Darcy Dataset.")
        kwargs["n_examples_per_sample"] = self.perm_example_xs.shape[1]

        super().__init__(input_size=(2,),
                         output_size=(2,),
                         total_n_functions=self.ys.shape[0],
                         total_n_samples_per_function=self.perm_example_ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=self.ys.shape[1],
                         *args, **kwargs,
                         )
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
        self.perm_example_xs = self.perm_example_xs.to(device)
        self.perm_example_ys = self.perm_example_ys.to(device)
        self.device = device

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,), device=self.device)
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        function_indicies = info["function_indicies"]
        if n_samples == self.n_examples_per_sample:
            xs = self.perm_example_xs[function_indicies]
        else:
            xs = self.xs[function_indicies]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        if inputs.shape[1] == self.n_examples_per_sample:
            ys = self.perm_example_ys[function_indicies]
        else:
            ys = self.ys[function_indicies]
        return ys

class LTgtDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/L-Shaped/linearDarcy_test.mat')
            tag = "u_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/L-Shaped/linearDarcy_train.mat')
            tag = "u_train"

        # fetch data
        x = mat['x']
        y = mat['y']
        u = mat[tag]

        # ins
        inputs = torch.tensor(np.concatenate([x,y], axis=1), dtype=torch.float32)
        self.xs = inputs.reshape(1, -1, 2).expand(u.shape[0], -1, -1)

        # outs
        self.ys = torch.tensor(u[:, :, None], dtype=torch.float32)

        # normalization is required here as the ys are tiny.
        # hard coded constants for consistency between train/test
        std = 0.008392013609409332
        self.ys = self.ys / std

        super().__init__(input_size=(2,),
                         output_size=(1,),
                         total_n_functions=self.ys.shape[0],
                         total_n_samples_per_function=self.ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=self.ys.shape[1],
                         *args, **kwargs,
                         )
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
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


def plot_positions_and_colors_with_smoothing(ax, xx, yy, intensity, image_density=200, vmin=None, vmax=None):
    # use cubic smoothing
    # creates grid
    xmin, xmax = xx.min().cpu().item(), xx.max().cpu().item()
    ymin, ymax = yy.min().cpu().item(), yy.max().cpu().item()
    grid_x, grid_y = np.mgrid[xmin:xmax:1j*image_density, ymin:ymax:1j*image_density]
    points = np.array([xx.flatten().cpu(), yy.flatten().cpu()]).T

    # smooth
    grid_intensity = griddata(points, intensity.cpu(), (grid_x, grid_y), method='cubic')

    # plot
    ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)

    # add a white square from 16/31 to 31/31, as this is the patch that is all zeros
    ax.add_patch(plt.Rectangle((16/31, 16/31), 15/31, 15/31, fill=True, color='white'))


def plot_source_L(xs, ys, y_hats, info, logdir):
    # plot
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    for row in range(4):
        # first col is ys[:, :, 0]
        ax = axs[row, 0]
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], ys[row, :, 0])
        ax.set_title("Groundtruth f1")

        # second col is y_hats[:, :, 0]
        ax = axs[row, 1]
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], y_hats[row, :, 0])
        ax.set_title("Estimate f1")

        # third col is ys[:, :, 1]
        ax = axs[row, 2]
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], ys[row, :, 1])
        ax.set_title("Groundtruth f2")

        # fourth col is y_hats[:, :, 1]
        ax = axs[row, 3]
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], y_hats[row, :, 1])
        ax.set_title("Estimate f2")

    plt.tight_layout()
    plt.savefig(f"{logdir}/source.png")
    plt.clf()

def plot_target_L(xs, ys, y_hats, info, logdir):
    # plot
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    for row in range(4):
        # first col is ys[:, :, 0]
        ax = axs[row, 0]
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], ys[row, :, 0])
        ax.set_title("Groundtruth u")

        # second col is y_hats[:, :, 0]
        ax = axs[row, 1]
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], y_hats[row, :, 0])
        ax.set_title("Estimate u")

    plt.tight_layout()
    plt.savefig(f"{logdir}/target.png")
    plt.clf()


def plot_transformation_L(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    size = 5
    model_type = info["model_type"]
    color = colors[model_type]
    label = labels[model_type]

    for row in range(example_xs.shape[0]):
        fig = plt.figure(figsize=(4.6 * size, 1 * size), dpi=300)
        # gridspec = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.12, 1, 1, 0.12])
        gridspec = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.12])
        axs = gridspec.subplots()
        
        # first col is ys[:, :, 0]
        ax = axs[0]
        # vmin = min(example_ys[row, :, 0].min().cpu().item(), example_ys[row, :, 0].min().cpu().item())
        # vmax = max(example_ys[row, :, 0].max().cpu().item(), example_ys[row, :, 0].max().cpu().item())
        plot_positions_and_colors_with_smoothing(ax, example_xs[row, :, 0], example_xs[row, :, 1], example_ys[row, :, 0])
        ax.set_title("$f_1$", fontsize=20)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])

        # second col is ys[:, :, 1]
        ax = axs[1]
        plot_positions_and_colors_with_smoothing(ax, example_xs[row, :, 0], example_xs[row, :, 1], example_ys[row, :, 1])
        ax.set_title("$f_2$", fontsize=20)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([])

        # create a color bar
        # ax = axs[2]
        # mappable = plt.cm.ScalarMappable(cmap='jet')
        # mappable.set_array([vmin, vmax])
        # plt.colorbar(mappable, cax=ax)
        

        # # add an arrow to the middle column
        # # and a T right above it
        # ax = axs[row, 2]
        # ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        # ax.text(0.1, 0.1, "T", fontsize=30)
        # ax.set_xlim(0, 0.5)
        # ax.set_ylim(-0.3, 0.3)
        # ax.axis("off")

        # this column is ys[:, :, 0]
        ax = axs[2]
        vmin = min(ys[row, :, 0].min().cpu().item(), y_hats[row, :, 0].min().cpu().item())
        vmax = max(ys[row, :, 0].max().cpu().item(), y_hats[row, :, 0].max().cpu().item())
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], ys[row, :, 0], vmin=vmin, vmax=vmax)
        ax.set_title("$u$", fontsize=20)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([])

        # this column is y_hats[:, :, 0]
        ax = axs[3]
        plot_positions_and_colors_with_smoothing(ax, xs[row, :, 0], xs[row, :, 1], y_hats[row, :, 0], vmin=vmin, vmax=vmax)
        ax.set_title("$\hat{u}$", fontsize=20)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([])

        # create a color bar
        ax = axs[4]
        mappable = plt.cm.ScalarMappable(cmap='jet')
        mappable.set_array([vmin, vmax])
        plt.colorbar(mappable, cax=ax)

        plt.tight_layout()
        plot_name = f"{logdir}/qualitative_LShaped_{label.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}_{row}.png"
        plt.savefig(plot_name)
        plt.clf()