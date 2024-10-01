import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from plotting_specs import colors, labels, titles


from src.Datasets.OperatorDataset import OperatorDataset

plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

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
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/Elastic/linearElasticity_test.mat')
            tag = "f_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/Elastic/linearElasticity_train.mat')
            tag = "f_train"
        # boundary force outputs
        f_bc = mat[tag]

        # boundaries are the f(x) outputs
        ys = f_bc

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
        self.xs = xs.to(device)
        self.ys = ys.to(device)
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


class ElasticPlateDisplacementDataset(OperatorDataset):
    def __init__(self, test=False, device="cuda", *args, **kwargs):
        # load data
        if test:
            mat = scipy.io.loadmat('./src/Datasets/Elastic/linearElasticity_test.mat')
            tags = "ux_test", "uy_test"
        else:
            mat = scipy.io.loadmat('./src/Datasets/Elastic/linearElasticity_train.mat')
            tags = "ux_train", "uy_train"

        # output data
        ux = mat[tags[0]]
        uy = mat[tags[0]]

        # input data
        xx = mat['x']
        yy = mat['y']

        # ys are the concatanated displacements (ux, uy)
        ys = np.concatenate((ux[:, :, None], uy[:, :, None]), axis=2)

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

        # normalize so data is in a reasonable range
        # these values are saved so they are the same for both train and test.
        mean, std = (2.8366714104777202e-05, 4.263603113940917e-05)
        ys = (ys - mean) / std # normalize

        self.xs = xs.to(device)
        self.ys = ys.to(device)
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
    fig = plt.figure(figsize=(4.6 * size, 1 * size), dpi=300)

    # create 3 types of plots
    gridspec_left = fig.add_gridspec(1, 3, )
    gridspec_cb1 = fig.add_gridspec(1, 1, )
    gridspec_right = fig.add_gridspec(1, 1, )
    gridspec_cb2 = fig.add_gridspec(1, 1, )
    
    # compute boundaries. 
    width_ratios=[1, 1, 1, 0.11, 1, 0.11]
    start = 0.05
    stop = 0.95
    wspace = 0.01
    available_space = stop - start - wspace * (len(width_ratios) - 1)
    width = available_space / sum(width_ratios)

    left1 = start
    right1 = start + width * 3
    left2 = right1 + wspace
    right2 = left2 + 0.11 * width
    left3 = right2 + 4.5 * wspace
    right3 = left3 + width
    left4 = right3 + wspace * 0.0
    right4 = left4 + 0.11 * width

    gridspec_left.update(left=left1, right=right1, wspace=0.15)
    gridspec_cb1.update(left=left2, right=right2)
    gridspec_right.update(left=left3, right=right3)
    gridspec_cb2.update(left=left4, right=right4)


    # adjust the horizontal spacing of the last plot
    

    model_type = info["model_type"]
    color = colors[model_type]
    label = labels[model_type]

    for row in range(example_xs.shape[0]):
        # plot the forcing function
        ax = fig.add_subplot(gridspec_left[0, 0])
        ax.plot(example_xs[row].cpu(), example_ys[row].cpu(), label="Groundtruth", color='black')
        if example_y_hats is not None:
            ax.plot(example_xs[row].cpu(), example_y_hats[row].cpu(), label=label, color=color)
        ax.set_title(f"Forcing function", fontsize=20)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.legend(frameon=False)
        ax.set_box_aspect(1)

        # plot the arrows for transformation
        # add an arrow to the middle column
        # and a T right above it
        # ax = fig.add_subplot(gridspec[row, 1])
        # ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        # ax.text(0.1, 0.1, "T", fontsize=30)
        # ax.set_xlim(0, 0.5)
        # ax.set_ylim(-0.3, 0.3)
        # ax.axis("off")
    
        # plot the transformation. Its the same 3d mesh plot as the target
        vmin = ys[row].min().item()
        vmax = ys[row].max().item()

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
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                distance = np.sqrt((grid_x[i, j] - 0.5) ** 2 + (grid_y[i, j] - 0.5) ** 2) # .min()
                if distance < 0.25:
                    grid_intensity[i, j] = np.nan

        # mesh plot
        ax = fig.add_subplot(gridspec_left[0, 1])
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        mesh = ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto', vmax=vmax, vmin=vmin)
        title = "Groundtruth X Displacement"
        ax.set_title(title, fontsize=20)
        ax.set_box_aspect(1)

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
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                distance = np.sqrt((grid_x[i, j] - 0.5) ** 2 + (grid_y[i, j] - 0.5) ** 2) # .min()
                if distance < 0.25:
                    grid_intensity[i, j] = np.nan

        # mesh plot
        ax = fig.add_subplot(gridspec_left[0, 2])
        mesh = ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto', vmax=vmax, vmin=vmin)
        title = "Predicted X Displacement"
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_title(title, fontsize=20)
        ax.set_box_aspect(1)


        # plot the colorbar
        ax = fig.add_subplot(gridspec_cb1[0, 0])
        cbar = plt.colorbar(mesh, cax=ax)
        # adjust the colorbar to set the labels to the left
        # cbar.ax.set_position(cbar.ax.get_position().translated(-0.05, 0))        
        # cbar.ax.yaxis.set_ticks_position('left')
        # cbar.ax.yaxis.set_label_position('left')
        ax.set_yticks([vmin,(vmax+vmin)/2 ,vmax])

        # now compute the difference between y_hats and y
        # then plot the same way
        # fetch data
        xx, yy = xs[row, :, 0].cpu(), xs[row, :, 1].cpu()
        groundtruth_displacement_x = ys[row, :, 0]
        predicted_displacement_x = y_hats[row, :, 0]
        difference = (groundtruth_displacement_x - predicted_displacement_x).abs()

        # plot details
        image_density = 200j
        grid_x, grid_y = np.mgrid[-xx.min().cpu().item():xx.max().cpu().item():image_density, 
                                  -yy.min().cpu().item():yy.max().item():image_density]
        points = np.array([xx.flatten().cpu(), yy.flatten().cpu()]).T


        # get colors
        intensity = difference
        grid_intensity = griddata(points, intensity.cpu(), (grid_x, grid_y), method='cubic')

        # remove the points inside the hole
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                distance = np.sqrt((grid_x[i, j] - 0.5) ** 2 + (grid_y[i, j] - 0.5) ** 2) # .min()
                if distance < 0.25:
                    grid_intensity[i, j] = np.nan



        # mesh plot
        vmax = (vmax - vmin) * ( 0.01)
        vmin = 0
        ax = fig.add_subplot(gridspec_right[0, 0])
        mesh = ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto', vmax=vmax, vmin=vmin)
        title = "Absolute Error"
        ax.set_title(title, fontsize=20)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_box_aspect(1)

        
        # plot the colorbar
        ax = fig.add_subplot(gridspec_cb2[0, 0])
        cbar = plt.colorbar(mesh, cax=ax)
        # shift it to the left by 0.05
        # print(cbar.ax.get_position())
        # cbar.ax.set_position(cbar.ax.get_position().translated(-0.15, 0))

        # adjust the colorbar to set the labels to the left
        # cbar.ax.set_position(cbar.ax.get_position().translated(-0.05, 0))        
        # cbar.ax.yaxis.set_ticks_position('left')
        # cbar.ax.yaxis.set_label_position('left')
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        
        

        plt.tight_layout(w_pad=0.2)
        plot_name = f"{logdir}/qualitative_Elastic_{label.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}_{row}.png"
        plt.savefig(plot_name)
        print("Saving to ", plot_name)
        plt.clf()


