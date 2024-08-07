import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata


from src.Datasets.OperatorDataset import OperatorDataset

# mat = scipy.io.loadmat('./src/Datasets/Dataset_1Circle.mat')
#
# # F(x) inputs
# f_bc_test = mat['f_bc_test']
# f_bc_train = mat['f_bc_train']
#
# # F(x) outputs
# stressX_train = mat['stressX_train']
# stressY_train = mat['stressY_train']
# stressX_test = mat['stressX_test']
# stressY_test = mat['stressY_test']
#
#
# # TF(x) outputs
# ux_train = mat['ux_train']
# uy_train = mat['uy_train']
#
# ux_test = mat['ux_test']
# uy_test = mat['uy_test']
#
#
# # x inputs
# xx = mat['xx']
# yy = mat['yy']
#
# print("xx shape: ", xx.shape)
# print("ux shape: ", ux_train.shape)
# print("stress x shape: ", stressX_train.shape)
# print("f_bc_train shape: ", f_bc_train.shape)




class ElasticPlateBoudaryForceDataset(OperatorDataset):
    def __init__(self, test=False, *args, **kwargs):
        # load data
        mat = scipy.io.loadmat('./src/Datasets/Elastic/Dataset_1Circle.mat')
            
        # boundary force outputs
        f_bc_test = mat['f_bc_test']
        f_bc_train = mat['f_bc_train']

        # boundaries are the f(x) outputs
        ys = f_bc_test if test else f_bc_train

        # xs are evenly spaced between 0 and 1
        xs = torch.linspace(0, 1, ys.shape[1])

        # format xs and ys
        xs = xs[None, :, None].repeat(ys.shape[0], 1, 1)
        ys = torch.tensor(ys).float().unsqueeze(2)

        if "n_examples_per_sample" in kwargs and kwargs["n_examples_per_sample"] != ys.shape[1]:
            print(f"WARNING: n_examples_per_sample is hard set to {ys.shape[1]} for the Elastic Dataset.")
        kwargs["n_examples_per_sample"] = ys.shape[1]

        
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                         *args, **kwargs,
        )
        self.xs = xs
        self.ys = ys

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


class ElasticPlateDisplacementDataset(OperatorDataset):
    def __init__(self, test=False, *args, **kwargs):
        # load data
        mat = scipy.io.loadmat('./src/Datasets/Elastic/Dataset_1Circle.mat')

        # output data
        ux_train = mat['ux_train']
        uy_train = mat['uy_train']

        ux_test = mat['ux_test']
        uy_test = mat['uy_test']

        # input data
        xx = mat['xx']
        yy = mat['yy']

        # ys are the concatanated displacements (ux, uy)
        if test:
            ys = np.concatenate((ux_test[:, :, None], uy_test[:, :, None]), axis=2)
        else:
            ys = np.concatenate((ux_train[:, :, None], uy_train[:, :, None]), axis=2)

        # xs are the concatanated x and y coordinates
        xs = np.concatenate((xx, yy), axis=1)
        xs = np.repeat(xs[None, :, :], ys.shape[0], axis=0)

        # format xs and ys
        xs = torch.tensor(xs).float()
        ys = torch.tensor(ys).float()

        super().__init__(input_size=(2,),
                         output_size=(2,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_points_per_sample=ys.shape[1],
                            *args, **kwargs,
        )

        mean = 0.000014114736 # these values are saved so they are the same for both train and test.
        std = 0.000037430502
        ys = (ys - mean) / std # normalize
        # print(f"Mean: {ys.mean():0.12f}, Std: {ys.std():0.12f}, max: {ys.max():0.6f}, min: {ys.min():0.6f}")
        # err
        self.xs = xs
        self.ys = ys 

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




def plot_source_boundary_force(xs, ys, y_hats, info, logdir):
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
        if i == 8:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{logdir}/source.png")
    plt.clf()

def plot_target_boundary(xs, ys, y_hats, info, logdir):

    # 4 rows, 5 cols. Last col is only 0.2 wide as it is a colorbar.
    # fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    fig = plt.figure(figsize=(4.4 * 5, 4.5 * 4), dpi=300)
    gridspec = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.2])

    # v_min and vmax based on data
    vmin = ys[0:4].min().item()
    vmax = ys[0:4].max().item()

    # each row is 1 function
    # first 2 cols are groundtruth x,y displacements
    # last 2 cols are estimated x,y displacements

    for row in range(4):

        # get input
        xx, yy = xs[row, :, 0].cpu(), xs[row, :, 1].cpu()

        # get output data
        groundtruth_displacement_x = ys[row, :, 0]
        groundtruth_displacement_y = ys[row, :, 1]
        predicted_displacement_x = y_hats[row, :, 0]
        predicted_displacement_y = y_hats[row, :, 1]


        # plot details
        image_density = 200j
        grid_x, grid_y = np.mgrid[-xx.min().cpu().item():xx.max().cpu().item():image_density, 
                                  -yy.min().cpu().item():yy.max().cpu().item():image_density]
        points = np.array([xx.flatten().cpu(), yy.flatten().cpu()]).T

        for col in range(4):
            intensity = groundtruth_displacement_x if col == 0 else \
                        groundtruth_displacement_y if col == 1 else \
                        predicted_displacement_x if col == 2 else \
                        predicted_displacement_y
            grid_intensity = griddata(points, intensity.cpu(), (grid_x, grid_y), method='cubic')

            # remove the points inside the hole
            max_distance = (xx.max() - xx.min()) / 10
            for i in range(grid_x.shape[0]):
                for j in range(grid_x.shape[1]):
                    distance = np.sqrt((grid_x[i, j] - xx) ** 2 + (grid_y[i, j] - yy) ** 2).min()
                    if distance > max_distance:
                        grid_intensity[i, j] = np.nan

            # mesh plot
            ax = fig.add_subplot(gridspec[row, col])
            mesh = ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto')

            # save
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel("x")
            plt.ylabel("y")
            title = "Groundtruth x displacement" if col == 0 else \
                    "Groundtruth y displacement" if col == 1 else \
                    "Predicted x displacement" if col == 2 else \
                    "Predicted y displacement"
            ax.set_title(title)

    # add color bar
    ax = fig.add_subplot(gridspec[:, 4])
    cbar = plt.colorbar(mesh, cax=ax)

    plt.tight_layout()
    plt.savefig(f"{logdir}/target.png")
    plt.clf()

def plot_transformation_elastic(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    size = 5
    fig = plt.figure(figsize=(3.4 * size, 2.5 * size), dpi=300)
    gridspec = fig.add_gridspec(4, 5, width_ratios=[1, 0.4, 1, 1, 0.2])
    # axs = gridspec.subplots()

    # plot the first 4 functions
    for row in range(4):
        ax = fig.add_subplot(gridspec[row, 0])
        ax.plot(example_xs[row].cpu(), example_ys[row].cpu(), label="Groundtruth")
        if example_y_hats is not None:
            ax.plot(example_xs[row].cpu(), example_y_hats[row].cpu(), label="Estimate")
        ax.set_title(f"Force function {row}")
        if row == 3:
            ax.legend()

    # plot the arrows for transformation
    for row in range(4):
        # add an arrow to the middle column
        # and a T right above it
        ax = fig.add_subplot(gridspec[row, 1])
        ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        ax.text(0.1, 0.1, "T", fontsize=30)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.axis("off")
    
    # plot the transformation. Its the same 3d mesh plot as the target
    vmin = ys[0:4].min().item()
    vmax = ys[0:4].max().item()
    for row in range(4):       
        # fetch data
        xx, yy = xs[row, :, 0].cpu(), xs[row, :, 1].cpu()
        groundtruth_displacement_x = ys[row, :, 0]
        
        # plot details
        image_density = 200j
        grid_x, grid_y = np.mgrid[-xx.min().cpu().item():xx.max().cpu().item():image_density, 
                                  -yy.min().cpu().item():yy.max().cpu().item():image_density]
        points = np.array([xx.flatten().cpu(), yy.flatten().cpu()]).T


        # get colors
        intensity = groundtruth_displacement_x
        grid_intensity = griddata(points, intensity.cpu(), (grid_x, grid_y), method='cubic')

        # remove the points inside the hole
        max_distance = (xx.max() - xx.min()) / 10
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                distance = np.sqrt((grid_x[i, j] - xx) ** 2 + (grid_y[i, j] - yy) ** 2).min()
                if distance > max_distance:
                    grid_intensity[i, j] = np.nan

        # mesh plot
        ax = fig.add_subplot(gridspec[row, 2])
        mesh = ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto')
        if row == 0:
            title = "Groundtruth x displacement"
            ax.set_title(title)

    for row in range(4):        
        # fetch data
        xx, yy = xs[row, :, 0].cpu(), xs[row, :, 1].cpu()
        estimated_displacement_x = y_hats[row, :, 0]
        
        # plot details
        image_density = 200j
        grid_x, grid_y = np.mgrid[-xx.min().cpu().item():xx.max().cpu().item():image_density, 
                                  -yy.min().cpu().item():yy.max().cpu().item():image_density]
        points = np.array([xx.flatten().cpu(), yy.flatten().cpu()]).T


        # get colors
        intensity = estimated_displacement_x
        grid_intensity = griddata(points, intensity.cpu(), (grid_x, grid_y), method='cubic')

        # remove the points inside the hole
        max_distance = (xx.max() - xx.min()) / 10
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                distance = np.sqrt((grid_x[i, j] - xx) ** 2 + (grid_y[i, j] - yy) ** 2).min()
                if distance > max_distance:
                    grid_intensity[i, j] = np.nan

        # mesh plot
        ax = fig.add_subplot(gridspec[row, 3])
        mesh = ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto')
        if row == 0:
            title = "Predicted x displacement"
            ax.set_title(title)


    # plot the colorbar
    ax = fig.add_subplot(gridspec[:, 4])
    cbar = plt.colorbar(mesh, cax=ax)

    plt.tight_layout()
    plt.savefig(f"{logdir}/transformation.png")
    plt.clf()

