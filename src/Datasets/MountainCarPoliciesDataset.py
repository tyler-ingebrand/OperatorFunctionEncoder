from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from matplotlib import pyplot as plt

from src.Datasets.OperatorDataset import OperatorDataset

f1 = lambda x: x[..., 0:1] ** 2
f2 = lambda x: x[..., 0:1] * x[..., 1:2]
f3 = lambda x: x[..., 1:2] ** 2
f4 = lambda x: torch.sin(3.5 * (x[..., 0:1] + 0.5))
f5 = lambda x: torch.sin(45 * (x[..., 1:2] - 0))
f6 = lambda x: torch.ones_like(x[..., 0:1])

class MountainCarPoliciesDataset(OperatorDataset):

    # all policies mapping mountain car states to actions between -1 and 1
    def __init__(self,
                 a_range=(-0,0),
                 b_range=(-0,0),
                 c_range=(-0,0),
                 d_range=(-1,1),
                 e_range=(-1,1),
                 f_range=(-0.2,0.2),
                 *args,
                 **kwargs,
                 ):
        super().__init__(input_size=(2,), output_size=(1,), *args, **kwargs)
        self.a_range = a_range # x ^ 2
        self.b_range = b_range # xy
        self.c_range = c_range # y ^ 2
        self.d_range = d_range # x
        self.e_range = e_range # y
        self.f_range = f_range # 1
        self.input_low = torch.tensor([-1.2, -0.07])
        self.input_high = torch.tensor([0.6, 0.07])




    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        As = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        Bs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        Cs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        Ds = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.d_range[1] - self.d_range[0]) + self.d_range[0]
        Es = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.e_range[1] - self.e_range[0]) + self.e_range[0]
        Fs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.f_range[1] - self.f_range[0]) + self.f_range[0]
        return {"As": As, "Bs": Bs, "Cs": Cs, "Ds": Ds, "Es": Es, "Fs": Fs}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        xs = torch.rand((info["As"].shape[0], n_samples, *self.input_size), dtype=torch.float32)
        xs = xs * (self.input_high - self.input_low) + self.input_low
        return xs

    @staticmethod
    def compute_outputs(info, inputs) -> torch.tensor:
        As, Bs, Cs, Ds, Es, Fs = info["As"], info["Bs"], info["Cs"], info["Ds"], info["Es"], info["Fs"]
        if len(inputs.shape) == 3:
            As = As.unsqueeze(1)
            Bs = Bs.unsqueeze(1)
            Cs = Cs.unsqueeze(1)
            Ds = Ds.unsqueeze(1)
            Es = Es.unsqueeze(1)
            Fs = Fs.unsqueeze(1)
        ys =    As * f1(inputs) + \
                Bs * f2(inputs) + \
                Cs * f3(inputs) + \
                Ds * f4(inputs) + \
                Es * f5(inputs) + \
                Fs * f6(inputs)

        # print(ys.max(), ys.min())
        # err
        return ys


class MountainCarEpisodesDataset(OperatorDataset):
    # the space of all possible episodes given a policy
    # maps a time to a state
    def __init__(self,
                 a_range=(-0,0),
                 b_range=(-0,0),
                 c_range=(-0,0),
                 d_range=(-1,1),
                 e_range=(-1,1),
                 f_range=(-0.2,0.2),
                 max_time=200,
                 *args,
                 **kwargs,
                 ):
        if "n_examples_per_sample" in kwargs: # this is hard set to the len of the episode. 
            del kwargs["n_examples_per_sample"]
        super().__init__(input_size=(1,), output_size=(2,),
                         n_examples_per_sample=max_time,
                         n_points_per_sample=max_time,
                         *args, **kwargs)
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.d_range = d_range
        self.e_range = e_range
        self.f_range = f_range

        self.init_state = torch.tensor([-0.5, 0])
        self.max_time = max_time

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.power = 0.0015

    def sample_info(self) -> dict:
        # generate n_functions sets of coefficients
        As = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        Bs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        Cs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        Ds = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.d_range[1] - self.d_range[0]) + self.d_range[0]
        Es = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.e_range[1] - self.e_range[0]) + self.e_range[0]
        Fs = torch.rand((self.n_functions_per_sample, 1), dtype=torch.float32) * (self.f_range[1] - self.f_range[0]) + self.f_range[0]
        return {"As": As, "Bs": Bs, "Cs": Cs, "Ds": Ds, "Es": Es, "Fs": Fs}

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        xs = torch.arange(0, self.max_time).unsqueeze(0).unsqueeze(2).expand(info["As"].shape[0], n_samples, 1).to(torch.float32)
        xs = xs / self.max_time # this regularization helps all approaches converge
        return xs

    def compute_outputs(self, info, inputs) -> torch.tensor:
        As, Bs, Cs, Ds, Es, Fs = info["As"], info["Bs"], info["Cs"], info["Ds"], info["Es"], info["Fs"]
        states = self.trajectory(As, Bs, Cs, Ds, Es, Fs)
        return states

    def trajectory(self, As, Bs, Cs, Ds, Es, Fs):
        # initalize first state
        states = torch.zeros((self.n_functions_per_sample, self.max_time, 2), dtype=torch.float32)
        states[:, 0, :] = self.init_state
        info = {"As": As, "Bs": Bs, "Cs": Cs, "Ds": Ds, "Es": Es, "Fs": Fs}

        # compute trajectory following actions listed
        for t in range(1, self.max_time):
            s = states[:, t - 1, :]
            a = MountainCarPoliciesDataset.compute_outputs(info, s)
            ns = self.step(s, a)
            states[:, t, :] = ns
        return states

    def step(self, s, a):
        # mountain car dynamics
        position = s[:, 0]
        velocity = s[:, 1]
        force = torch.clamp(a[:, 0], self.min_action, self.max_action)

        velocity += force * self.power - 0.0025 * torch.cos(3 * position)
        velocity = torch.clamp(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = torch.clamp(position, self.min_position, self.max_position)
        velocity[torch.logical_and(position == self.min_position, velocity < 0)] = 0

        ns = torch.stack([position, velocity], dim=1)

        return ns

def plot_source_mountain_car(xs, ys, y_hats, info, logdir):
    # now plot comparisons. We plot the groundtruth on the left and the predicted on the right, 2 cols, 4 rows
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    for i in range(4):
        ax = axs[i, 0]
        ax.set_title(f"Groundtruth Policy {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.scatter(xs[i, :, 0].cpu(), xs[i, :, 1].cpu(), c=ys[i].cpu(), cmap="coolwarm", vmin=-1, vmax=1)

        ax = axs[i, 1]
        ax.set_title(f"Predicted Policy {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.scatter(xs[i, :, 0].cpu(), xs[i, :, 1].cpu(), c=y_hats[i].cpu(), cmap="coolwarm", vmin=-1, vmax=1)

    plt.savefig(f"{logdir}/source.png")

def plot_target_mountain_car(xs, ys, y_hats, info, logdir):
    # now plot comparisons. We plot the groundtruth on the left and the predicted on the right, 2 cols, 4 rows
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))

    for i in range(4):
        ax = axs[i, 0]
        ax.set_title(f"Groundtruth Trajectory {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.plot(ys[i][:, 0].cpu(), ys[i][:, 1].cpu())
        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-0.07, 0.07)

        ax = axs[i, 1]
        ax.set_title(f"Predicted Trajectory {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.plot(y_hats[i][:, 0].cpu(), y_hats[i][:, 1].cpu())
        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-0.07, 0.07)

    plt.savefig(f"{logdir}/target.png")


def plot_transformation_mountain_car(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir):
    size = 5


    for row in range(example_xs.shape[0]):
        # create plot
        fig = plt.figure(figsize=(2.6 * size, 1 * size), dpi=300)
        gridspec = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.4, 1])
        axs = gridspec.subplots()        
        
        # plot
        ax = axs[0]
        ax.set_title(f"Predicted Policy {row}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        
        # create a cmap with -1,1 as the min and max
        cmap = plt.cm.coolwarm
        if example_y_hats is not None:
            ax.scatter(example_xs[row, :, 0].cpu(), example_xs[row,:, 1].cpu(), c=example_y_hats[row].cpu(), cmap=cmap, vmin=-1, vmax=1)
        else:
            ax.scatter(example_xs[row, :, 0].cpu(), example_xs[row,:, 1].cpu(), c=example_ys[row].cpu(), cmap=cmap, vmin=-1, vmax=1)
        
        # plot the colorbar
        # make the labels -1=left, 0=stay, 1=right
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Left', 'Idle', 'Right'])


        # add an arrow to the middle column
        # and a T right above it
        ax = axs[1]
        ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        ax.text(0.1, 0.1, "T", fontsize=30)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.axis("off")

        # plot
        ax = axs[2]
        ax.set_title(f"Predicted Trajectory {row}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.plot(y_hats[row][:, 0].cpu(), y_hats[row][:, 1].cpu(), label="Estimated")
        ax.plot(ys[row][:, 0].cpu(), ys[row][:, 1].cpu(), label="Groundtruth")
        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-0.07, 0.07)
        if row == 3:
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"{logdir}/transformation_{row}.png")
