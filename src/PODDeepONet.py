
import torch
from FunctionEncoder import BaseDataset, BaseCallback
from tqdm import trange

from src.Datasets.OperatorDataset import OperatorDataset


# This implements an unstacked DeepONet
class DeepONet_POD(torch.nn.Module):

    def __init__(self,
                 input_size_src, # the dimensionality of the inputs to u. In the paper it is always 1, but we can be more general.
                 output_size_src, # the dimensionality of the output of u. In the paper it is always 1, but we can be more general.
                 input_size_tgt, # the dimensionality y. In the paper it is always 1, but we can be more general.
                 output_size_tgt, # the dimensionality of the output of y. In the paper it is always 1, but we can be more general.
                 n_input_sensors, # the number of input sensors, "m" in the paper
                 p=20, # This is the number of terms for the final dot product operation. In the paper, they say at least 10.
                 use_deeponet_bias=True, # whether to use a bias term after the cross product in the deepnet.
                 hidden_size=256,
                 n_layers=4,
                 ):
        super().__init__()

        # set hyperparameters
        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt

        self.n_input_sensors = n_input_sensors
        self.p = p
        self.hidden_size = hidden_size

        # this maps u(x_1), u(x_2), ..., u(x_m) to b_1, b_2, ..., b_p
        layers_branch = []
        layers_branch.append(torch.nn.Linear(output_size_src * n_input_sensors, hidden_size))
        layers_branch.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            layers_branch.append(torch.nn.Linear(hidden_size, hidden_size))
            layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.Linear(hidden_size, output_size_tgt * p))
        self.branch = torch.nn.Sequential(*layers_branch)

        # this maps y to t_1, ..., t_p
        # the trunk basis is computed via POD
        self.trunk = None

        # create optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        # holdovers from function encoder code, these do nothing
        self.method = "deepONet_pod"
        self.average_function = None

    def compute_POD(self, dataset: OperatorDataset):
        # see https://github.com/lu-group/deeponet-fno/blob/main/src/darcy_rectangular_pwc/deeponet_POD.py

        with torch.no_grad():
            # set device
            device = next(self.parameters()).device

            # This data is drawn from the target function space ONLY
            _, _, _, us, _ = dataset.sample(device=device)

            # us is of size n_functions, n_points, dimensionality
            # We will solve each dimensionality independently.

            # first subtract the mean of each column
            means = us.mean(dim=0, keepdim=True)
            us = us - means

            # compute the covariance matrix
            covariance = torch.einsum("mfd,fnd->mnd", us.transpose(0,1), us) * (1/us.shape[0])
            covariance = covariance.permute(2,0,1)

            # compute eigen decomp
            # returns eigen values, vectors
            w, v = torch.linalg.eigh(covariance)

            # assert decomp worked, e.g. Av = wv
            covariance_approx = v @ torch.diag_embed(w) @ v.mT
            print("Recreation accuracy:" , torch.dist(covariance, covariance_approx))

            # flip to get most important eigen vectors
            w = w.flip(dims=(1,))
            v_flipped = v.flip(dims=(2,)) # this is a batch version of np.fliplr()

            # store the basis
            v_flipped *= means.shape[1] ** 0.5

            self.mean = means
            self.trunk = v_flipped.permute(1, 2, 0)[:, :self.p, :] # the largest p eigen vectors

    def forward_branch(self, u):
        ins = u.reshape(u.shape[0], -1)
        outs = self.branch(ins)
        outs = outs.reshape(outs.shape[0], -1, self.output_size_tgt)
        return outs

    def forward_trunk(self, y):
        with torch.no_grad():
            return self.trunk

    def forward(self, xs, us, ys):
        # xs are not actually used for deeponet, but we keep them to be consistent with the function encoder
        # us are the values of u at the input sensors
        # ys are the locations of the output sensors.
        b = self.forward_branch(us)
        t = self.forward_trunk(ys)

        # this is just the dot product, but allowing for the output dim to be > 1
        G_u_y = torch.einsum("fpz,dpz->fdz", b, t)
        G_u_y = G_u_y + self.mean

        return G_u_y

    # This is the main training loop, kept consistent with the function encoder code.
    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback: BaseCallback = None):
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())


        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            # sample input data
            xs, u_xs, ys, G_u_ys, _ = dataset.sample(device=device)


            # approximate functions, compute error
            estimated_G_u_ys = self.forward(xs, u_xs, ys)
            prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)

            # add loss components
            loss = prediction_loss

            # backprop with gradient clipping
            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())

    def _param_string(self):
        """ Returns a dictionary of hyperparameters for logging."""
        params = {}
        params["input_size_src"] = self.input_size_src
        params["output_size_src"] = self.output_size_src
        params["input_size_tgt"] = self.input_size_tgt
        params["output_size_tgt"] = self.output_size_tgt
        params["n_input_sensors"] = self.n_input_sensors
        params["p"] = self.p
        params["hidden_size"] = self.hidden_size
        params = {k: str(v) for k, v in params.items()}
        return params

