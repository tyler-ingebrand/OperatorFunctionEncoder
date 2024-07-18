import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata


from src.Datasets.OperatorDataset import OperatorDataset

# see https://github.com/mdribeiro/DeepCFD/tree/master?tab=readme-ov-file

# load src/Datasets/FluidFlow/dataX.pkl
# dataX = np.load('./src/Datasets/FluidFlow/dataX.pkl', allow_pickle=True)
# dataY = np.load('./src/Datasets/FluidFlow/dataY.pkl', allow_pickle=True)
#
# print(dataX.shape)
# print(dataY.shape)
#
#
# # check if the third channel ever varies. It doesnt.
# first = dataX[0, 2]
# for i in range(1, dataX.shape[0]):
#     if not np.allclose(first, dataX[i, 2]):
#         print("Third channel varies")
#         break
#
# sample = 1
# img1 = dataX[sample, 0]
# img2 = dataX[sample, 1]
# img3 = dataX[sample, 2]
# vmin, vmax = dataX[sample].min(), dataX[sample].max()
#
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].imshow(img1, cmap='jet', vmin=0)
# axs[1].imshow(img2, cmap='jet',)# vmin=vmin, vmax=vmax)
# axs[2].imshow(img3, cmap='jet',)# vmin=vmin, vmax=vmax)
# plt.savefig("input.png")
#
# img1 = dataY[sample, 0]
# img2 = dataY[sample, 1]
# img3 = dataY[sample, 2]
# vmin, vmax = dataY[sample].min(), dataY[sample].max()
#
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].imshow(img1, cmap='jet',)
# axs[1].imshow(img2, cmap='jet',) #vmin=vmin, vmax=vmax)
# axs[2].imshow(img3, cmap='jet',)# vmin=vmin, vmax=vmax)
# plt.savefig("output.png")




# see https://github.com/mdribeiro/DeepCFD/tree/master?tab=readme-ov-file
class FluidBoundaryDataset(OperatorDataset):
    def __init__(self, test=False, *args, **kwargs):
        # load data
        # dataX is the source function space
        if test:
            dataX = np.load('./src/Datasets/FluidFlow/test_dataX.npy', allow_pickle=True)
        else:
            dataX = np.load('./src/Datasets/FluidFlow/train_dataX.npy', allow_pickle=True)

        # this is distance to the object.
        # first dimension is n_functions.
        # second and third dims correspond to X,Y as in an image.
        ys = torch.tensor(dataX[:, 0, :, :])
        ys = ys.reshape(ys.shape[0], -1)
        ys = ys.unsqueeze(2)
        ys = torch.clamp(ys, min=0) # points inside obstacle are set to -50. This is discontinuous, so we set them to 0.
        # TODO: Do something smarter here so its continuous

        # this is the position coordinates of the above data, normalized to -1,1 range.
        xx1 = torch.linspace(-1, 1, dataX.shape[2])
        xx2 = torch.linspace(-1, 1, dataX.shape[3])
        xx1, xx2 = torch.meshgrid(xx1, xx2)
        xs = torch.stack((xx1.reshape(-1), xx2.reshape(-1)), dim=-1)
        xs = xs.repeat(ys.shape[0], 1, 1)


        super().__init__(input_size=(2,),
                         output_size=(1,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_examples_per_sample=1000,
                         n_points_per_sample=10_000,
                         *args, **kwargs,
        )
        self.xs = xs
        self.ys = ys
        self.xx1 = torch.linspace(-1, 1, dataX.shape[2])
        self.xx2 = torch.linspace(-1, 1, dataX.shape[3])
        self.sample_indicies = None

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,))
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        function_indicies = info["function_indicies"]
        if self.sample_indicies is None or not self.freeze_example_xs:
            self.sample_indicies = torch.randint(0, self.ys.shape[1], (n_samples,))

        xs = self.xs[function_indicies]
        xs = xs[:, self.sample_indicies]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        ys = self.ys[function_indicies]
        ys = ys[:, self.sample_indicies]
        return ys


class FluidVelocityDataset(OperatorDataset):
    def __init__(self, test=False):
        # load data
        # dataY is the target function space
        if test:
            dataY = np.load('./src/Datasets/FluidFlow/test_dataY.npy', allow_pickle=True)
        else:
            dataY = np.load('./src/Datasets/FluidFlow/train_dataY.npy', allow_pickle=True)


        # first dimension is n_functions.
        # second and third dims correspond to X,Y as in an image.
        ys = torch.tensor(dataY[:, :, :, :])
        ys = ys.transpose(1, 2).transpose(2,3)
        ys = ys.reshape(ys.shape[0], -1, ys.shape[3])
        # normalization
        norms = (0.05528882145881653, 0.01738288253545761, 0.011722780764102936)
        ys[:, :, 0] /= norms[0]
        ys[:, :, 1] /= norms[1]
        ys[:, :, 2] /= norms[2]
        # TODO: Do something smarter here so its continuous


        # this is the position coordinates of the above data, normalized to -1,1 range.
        xx1 = torch.linspace(-1, 1, dataY.shape[2])
        xx2 = torch.linspace(-1, 1, dataY.shape[3])
        xx1, xx2 = torch.meshgrid(xx1, xx2)
        xs = torch.stack((xx1.reshape(-1), xx2.reshape(-1)), dim=-1)
        xs = xs.repeat(ys.shape[0], 1, 1)



        super().__init__(input_size=(2,),
                         output_size=(3,),
                         total_n_functions=ys.shape[0],
                         total_n_samples_per_function=ys.shape[1],
                         n_functions_per_sample=10,
                         n_examples_per_sample=1000,
                         n_points_per_sample=10_000,
        )

        self.xs = xs
        self.ys = ys
        self.xx1 = torch.linspace(-1, 1, dataY.shape[2])
        self.xx2 = torch.linspace(-1, 1, dataY.shape[3])



    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        function_indicies = torch.randint(0, self.ys.shape[0], (self.n_functions_per_sample,))
        return {"function_indicies": function_indicies}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        function_indicies = info["function_indicies"]
        xs = self.xs[function_indicies]
        self.sample_indicies = torch.randint(0, self.ys.shape[1], (n_samples,))
        xs = xs[:, self.sample_indicies]
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        function_indicies = info["function_indicies"]
        ys = self.ys[function_indicies]
        ys = ys[:, self.sample_indicies]
        return ys




def plot_source_distance_to_object(xs, ys, y_hats, info, logdir):
    # now plot comparisons. We plot the groundtruth on the left and the predicted on the right, 2 cols, 4 rows
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    for i in range(4):
        ax = axs[i, 0]
        ax.set_title(f"Groundtruth Distances {i}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        intensity = ys[i].cpu()
        intensity[intensity<=0] = np.nan
        ax.scatter(xs[i, :, 0].cpu(), xs[i, :, 1].cpu(), c=intensity, cmap="coolwarm", vmin=0)

        ax = axs[i, 1]
        ax.set_title(f"Predicted Distances {i}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        intensity = y_hats[i].cpu()
        intensity[intensity<=0] = np.nan
        ax.scatter(xs[i, :, 0].cpu(), xs[i, :, 1].cpu(), c=intensity, cmap="coolwarm", vmin=0)

    plt.savefig(f"{logdir}/source.png")

def plot_target_fluid_flow(xs, ys, y_hats, info, logdir):

    # 3 rows, 3 cols. Last col is only 0.2 wide as it is colorbars.
    # fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    fig = plt.figure(figsize=(2.4 * 4, 2 * 4), dpi=300)
    gridspec = fig.add_gridspec(3, 2, width_ratios=[1, 1,])


    # first row is the ground truth and predicted velocity in x direction
    # ground truth
    ax = fig.add_subplot(gridspec[0, 0])
    ax.set_title(f"Groundtruth x velocity")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    v_min = min(ys[0, :, 0].min().item(), y_hats[0, :, 0].min().item())
    v_max = max(ys[0, :, 0].max().item(), y_hats[0, :, 0].max().item())
    ax.scatter(xs[0, :, 0].cpu(), xs[0, :, 1].cpu(), c=ys[0, :, 0].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

    # estimated
    ax = fig.add_subplot(gridspec[0, 1])
    ax.set_title(f"Predicted x velocity")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.scatter(xs[0, :, 0].cpu(), xs[0, :, 1].cpu(), c=y_hats[0, :, 0].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

    # colobar
    # cax = fig.add_subplot(gridspec[0, 2])
    # cbar = plt.colorbar(ax=cax, mappable=ax, orientation='vertical')

    # next row is y vel
    # ground truth
    ax = fig.add_subplot(gridspec[1, 0])
    ax.set_title(f"Groundtruth y velocity")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    v_min = min(ys[0, :, 1].min().item(), y_hats[0, :, 1].min().item())
    v_max = max(ys[0, :, 1].max().item(), y_hats[0, :, 1].max().item())
    ax.scatter(xs[0, :, 0].cpu(), xs[0, :, 1].cpu(), c=ys[0, :, 1].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

    # estimated
    ax = fig.add_subplot(gridspec[1, 1])
    ax.set_title(f"Predicted y velocity")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.scatter(xs[0, :, 0].cpu(), xs[0, :, 1].cpu(), c=y_hats[0, :, 1].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

    # colobar
    # cax = fig.add_subplot(gridspec[1, 2])
    # cbar = plt.colorbar(ax=cax, mappable=ax, orientation='vertical')

    # next row is pressure
    # ground truth
    ax = fig.add_subplot(gridspec[2, 0])
    ax.set_title(f"Groundtruth pressure")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    v_min = min(ys[0, :, 2].min().item(), y_hats[0, :, 2].min().item())
    v_max = max(ys[0, :, 2].max().item(), y_hats[0, :, 2].max().item())
    ax.scatter(xs[0, :, 0].cpu(), xs[0, :, 1].cpu(), c=ys[0, :, 2].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

    # estimated
    ax = fig.add_subplot(gridspec[2, 1])
    ax.set_title(f"Predicted pressure")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.scatter(xs[0, :, 0].cpu(), xs[0, :, 1].cpu(), c=y_hats[0, :, 2].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

    # colobar
    # cax = fig.add_subplot(gridspec[2, 2])
    # cbar = plt.colorbar(ax=cax, mappable=ax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(f"{logdir}/target.png")
    plt.clf()

def plot_transformation_fluid(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    size = 5
    fig = plt.figure(figsize=(7.4 * size, 4.5 * size), dpi=300)
    gridspec = fig.add_gridspec(4, 8, width_ratios=[1, 0.4, 1, 1, 1, 1, 1, 1])
    # axs = gridspec.subplots()

    # plot the first 4 functions
    for row in range(4):

        # first plot the source distance to object
        ax = fig.add_subplot(gridspec[row, 0])
        ax.set_title(f"Source Distance to Object {row}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        intensity = example_ys[row].cpu()
        intensity[intensity<=0] = np.nan
        ax.scatter(example_xs[row, :, 0].cpu(), example_xs[row, :, 1].cpu(), c=intensity, cmap="coolwarm", vmin=0)

        # add an arrow to the middle column
        # and a T right above it
        ax = fig.add_subplot(gridspec[row, 1])
        ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        ax.text(0.1, 0.1, "T", fontsize=30)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.axis("off")

        # then add groundtruth x velocity
        ax = fig.add_subplot(gridspec[row, 2])
        ax.set_title(f"Groundtruth x velocity {row}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        v_min = min(ys[row, :, 0].min().item(), y_hats[row, :, 0].min().item())
        v_max = max(ys[row, :, 0].max().item(), y_hats[row, :, 0].max().item())
        ax.scatter(xs[row, :, 0].cpu(), xs[row, :, 1].cpu(), c=ys[row, :, 0].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

        # then add predicted x velocity
        ax = fig.add_subplot(gridspec[row, 3])
        ax.set_title(f"Predicted x velocity {row}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.scatter(xs[row, :, 0].cpu(), xs[row, :, 1].cpu(), c=y_hats[row, :, 0].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

        # then add groundtruth y velocity
        ax = fig.add_subplot(gridspec[row, 4])
        ax.set_title(f"Groundtruth y velocity {row}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        v_min = min(ys[row, :, 1].min().item(), y_hats[row, :, 1].min().item())
        v_max = max(ys[row, :, 1].max().item(), y_hats[row, :, 1].max().item())
        ax.scatter(xs[row, :, 0].cpu(), xs[row, :, 1].cpu(), c=ys[row, :, 1].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

        # then add predicted y velocity
        ax = fig.add_subplot(gridspec[row, 5])
        ax.set_title(f"Predicted y velocity {row}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.scatter(xs[row, :, 0].cpu(), xs[row, :, 1].cpu(), c=y_hats[row, :, 1].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

        # then add groundtruth pressure
        ax = fig.add_subplot(gridspec[row, 6])
        ax.set_title(f"Groundtruth pressure {row}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        v_min = min(ys[row, :, 2].min().item(), y_hats[row, :, 2].min().item())
        v_max = max(ys[row, :, 2].max().item(), y_hats[row, :, 2].max().item())
        ax.scatter(xs[row, :, 0].cpu(), xs[row, :, 1].cpu(), c=ys[row, :, 2].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

        # then add predicted pressure
        ax = fig.add_subplot(gridspec[row, 7])
        ax.set_title(f"Predicted pressure {row}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.scatter(xs[row, :, 0].cpu(), xs[row, :, 1].cpu(), c=y_hats[row, :, 2].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)


    plt.tight_layout()
    plt.savefig(f"{logdir}/transformation.png")
    plt.clf()


