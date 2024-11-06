import torch
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
from plotting_specs import colors, labels, titles


from src.Datasets.OperatorDataset import OperatorDataset

if __name__ == "__main__":
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter

    # load ./src/Datasets/Burger/Burger.mat
    assert os.path.exists('./src/Datasets/Burger/Burger.mat'), "Burger.mat not found"
    dataset = loadmat('./src/Datasets/Burger/Burger.mat')
    inputs = dataset['input']
    outputs = dataset['output']
    tspan = dataset['tspan']
    sigma, tau = dataset['sigma'], dataset['tau']

    print(f"Time is discretized from {tspan[0, 0]} to {tspan[0, -1]} at {inputs.shape[1]} points")
    print(f"Position is discretized at {inputs.shape[1]} points")
    print("The input function is the initial wave state")
    print(f"The output function is the wave state over time")
    print(f"There are {inputs.shape[0]} examples in the dataset")
    print(f"sigma = {sigma[0][0]}, tau = {tau[0][0]}")


    # save it as a gif
    fig, ax = plt.subplots(1, 1)
    def update(i):
        time_index = i % inputs.shape[1]
        example_index = i // inputs.shape[1]
        ax.clear()
        if time_index == 0:
            ax.plot(inputs[example_index])
        else:
            ax.plot(outputs[example_index, time_index-1])
        ax.set_title(f"t = {tspan[0, time_index]}")
        return ax

    ani = animation.FuncAnimation(fig, update, frames=10 * inputs.shape[1], repeat=False)
    writer = PillowWriter(fps=30)
    ani.save("burger.gif", writer=writer)


class BurgerInputDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        mat = loadmat('./src/Datasets/Burger/Burger.mat')

        # grab the correct dimensions
        input_functions = mat['input']
        positions = torch.linspace(0, 1, input_functions.shape[1]).reshape(1, -1, 1)

        # grab training or testing data
        split = int(0.8 * input_functions.shape[0])
        if not test:
            input_functions = input_functions[:split]
        else:
            input_functions = input_functions[split:]


        # format xs and ys
        xs = positions
        ys = torch.tensor(input_functions).float().unsqueeze(2)
        kwargs["n_examples_per_sample"] = ys.shape[1]
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                         *args, **kwargs,
                         )

        # normalize. Xs are already between 0,1.
        # just need to normalize ys.
        mean, std = (1.1376314432709478e-05, 0.21469584107398987)
        ys = (ys - mean) / std

        self.xs = xs.to(device)
        self.ys = ys.to(device)
        self.device = device

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,), device=self.device)
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        xs = self.xs.repeat(self.n_functions_per_sample, 1, 1)
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        ys = self.ys[function_indicies]
        return ys


class BurgerOutputDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        mat = loadmat('./src/Datasets/Burger/Burger.mat')

        # grab the correct dimensions
        output_functions = mat['output']
        times = torch.tensor(mat['tspan'][0]).float()
        positions = torch.linspace(0, 1, output_functions.shape[2])


        # grab training or testing data
        split = int(0.8 * output_functions.shape[0])
        if not test:
            output_functions = output_functions[:split]
        else:
            output_functions = output_functions[split:]


        # format xs and ys
        # xs is the grid of positions and times
        # ys is the output functions
        # times has shape 1, 101 and position has shape 101
        X,Y = torch.meshgrid(positions, times.squeeze(0), indexing='ij')
        xs = torch.stack([X, Y], dim=-1)
        xs = xs.reshape(1, -1, 2)
        ys = torch.tensor(output_functions).float().reshape(output_functions.shape[0], -1, 1)

        super().__init__(input_size=(2,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                         *args, **kwargs,
                         )

        # normalize. Xs are already between 0,1.
        # just need to normalize ys.
        mean, std  =(-2.107516820615274e-06, 0.16743765771389008)
        ys = (ys - mean) / std

        self.sample_indicies = None
        self.xs = xs.to(device)
        self.ys = ys.to(device)
        self.device = device

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,), device=self.device)
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        xs = self.xs.repeat(self.n_functions_per_sample, 1, 1)
        # now sample a subset of the pos/time locations
        if not self.freeze_xs:
            self.sample_indicies = torch.randint(0, self.xs.shape[1], (n_samples,), device=self.device)
        elif self.sample_indicies is None:
            self.sample_indicies = torch.randint(0, self.xs.shape[1], (n_samples,), device=self.device)

        xs = xs[:, self.sample_indicies]
        # those indicies are stored for the output function
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        ys = self.ys[function_indicies][:, self.sample_indicies]
        return ys


def plot_source_burger(xs, ys, y_hats, info, logdir):

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



def plot_target_burger(xs, ys, y_hats, info, logdir):
    pass

def plot_transformation_burger(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    assert ys.shape == (ys.shape[0], 10201, 1)
    assert y_hats.shape == (ys.shape[0], 10201, 1)

    model_type = info["model_type"]
    color = colors[model_type]
    label = labels[model_type]
    size=5

    for i in range(ys.shape[0]):
        fig = plt.figure(figsize=(2.8 * size, 1 * size), dpi=300)

        # create 3 types of plots
        gridspec_left = fig.add_gridspec(1, 2, )
        gridspec_cb1 = fig.add_gridspec(1, 1, )
        gridspec_right = fig.add_gridspec(1, 1, )
        gridspec_cb2 = fig.add_gridspec(1, 1, )

        # compute boundaries.
        width_ratios = [1, 1, 0.11, 1, 0.11]
        start = 0.05
        stop = 0.93
        wspace = 0.01
        available_space = stop - start - wspace * (len(width_ratios) - 1)
        width = available_space / sum(width_ratios)

        left1 = start
        right1 = start + width * 2
        left2 = right1 + 2*wspace 
        right2 = left2 + 0.11 * width
        left3 = right2 + 6.5 * wspace
        right3 = left3 + width
        left4 = right3 + wspace
        right4 = left4 + 0.11 * width

        gridspec_left.update(left=left1, right=right1, wspace=0.17)
        gridspec_cb1.update(left=left2, right=right2)
        gridspec_right.update(left=left3, right=right3)
        gridspec_cb2.update(left=left4, right=right4)

        # plot ground truth
        ax = fig.add_subplot(gridspec_left[0])
        ys2 = ys[i].flatten().reshape(101, 101)
        xs2 = xs[i].flatten().reshape(101, 101, 2)
        vmin, vmax = ys2.min().cpu().numpy(), ys2.max().cpu().numpy()
        ax.scatter(xs2[:,:,0].cpu().numpy(), xs2[:,:,1].cpu().numpy(), c=ys2.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title("Ground Truth")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # plot estimate
        ax = fig.add_subplot(gridspec_left[1])
        y_hats2 = y_hats[i].flatten().reshape(101, 101)
        ax.scatter(xs2[:,:,0].cpu().numpy(), xs2[:,:,1].cpu().numpy(), c=y_hats2.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"{label} Estimate")
        ax.set_xlabel("Time")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # plot a color bar
        ax = fig.add_subplot(gridspec_cb1[0])
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        ax.set_title("Velocity")
        cbar = fig.colorbar(sm, cax=ax)
        cbar.locator = plt.MaxNLocator(5)
        cbar.update_ticks()

        # plot abs difference
        ax = fig.add_subplot(gridspec_right[0])
        vmax = (vmax - vmin) * 0.05
        vmin = 0
        diff = torch.abs(ys2 - y_hats2).cpu().numpy()
        ax.scatter(xs2[:,:,0].cpu().numpy(), xs2[:,:,1].cpu().numpy(), c=diff, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title("Absolute Difference")
        ax.set_xlabel("Time")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # plot a color bar with 5 ticks
        ax = fig.add_subplot(gridspec_cb2[0])
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm, )
        sm.set_array([])
        ax.set_title("Velocity")
        cbar = fig.colorbar(sm, cax=ax)
        cbar.locator = plt.MaxNLocator(5)
        cbar.update_ticks()

        # save
        # plt.tight_layout()
        print("Saving to ", f"{logdir}/transformation_{i}.png")
        plt.savefig(f"{logdir}/transformation_{i}.png", bbox_inches='tight')
        plt.clf()







