
import torch
from FunctionEncoder import BaseDataset, BaseCallback
from tqdm import trange


# This implements an unstacked DeepONet
class DeepONet_CNN(torch.nn.Module):

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
        # this becomes a CNN
        # the conv layers
        layers_branch = []
        layers_branch.append(torch.nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1))
        layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers_branch.append(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
        layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers_branch.append(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.Flatten())
        # mlp layers
        layers_branch.append(torch.nn.Linear(3136, hidden_size))
        layers_branch.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            layers_branch.append(torch.nn.Linear(hidden_size, hidden_size))
            layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.Linear(hidden_size, output_size_tgt * p))
        self.branch = torch.nn.Sequential(*layers_branch)

        # this maps y to t_1, ..., t_p
        # this is the exact same as MLP deeponet
        trunk_layers = []
        trunk_layers.append(torch.nn.Linear(input_size_tgt, hidden_size))
        trunk_layers.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            trunk_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(torch.nn.ReLU())
        trunk_layers.append(torch.nn.Linear(hidden_size, output_size_tgt * p))
        trunk_layers.append(torch.nn.Sigmoid())
        self.trunk = torch.nn.Sequential(*trunk_layers)

        # an optional bias, see equation 2 in the paper.
        self.bias = torch.nn.Parameter(torch.randn(output_size_tgt) * 0.1) if use_deeponet_bias else None

        # create optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        # holdovers from function encoder code, these do nothing
        self.method = "deepONet_cnn"
        self.average_function = None

    def forward_branch(self, u):
        ins = u.reshape(u.shape[0], 31, 31, u.shape[-1])
        ins = ins.permute(0, 3, 1, 2)
        outs = self.branch(ins)
        outs = outs.reshape(outs.shape[0], -1, self.output_size_tgt)
        return outs

    def forward_trunk(self, y):
        outs = self.trunk(y)
        outs = outs.reshape(outs.shape[0], y.shape[1], -1, self.output_size_tgt)
        return outs

    def forward(self, xs, us, ys):
        # xs are not actually used for deeponet, but we keep them to be consistent with the function encoder
        # us are the values of u at the input sensors
        # ys are the locations of the output sensors.
        b = self.forward_branch(us)
        t = self.forward_trunk(ys)

        # this is just the dot product, but allowing for the output dim to be > 1
        G_u_y = torch.einsum("fpz,fdpz->fdz", b, t)

        # optionally add bias
        if self.bias is not None:
            G_u_y = G_u_y + self.bias

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
        params["use_deeponet_bias"] = self.bias is not None
        params = {k: str(v) for k, v in params.items()}
        return params

class DeepONet_2Stage_CNN_branch(torch.nn.Module):

    def __init__(self,
                input_size_src, # the dimensionality of the inputs to u. In the paper it is always 1, but we can be more general.
                output_size_src, # the dimensionality of the output of u. In the paper it is always 1, but we can be more general.
                input_size_tgt, # the dimensionality y. In the paper it is always 1, but we can be more general.
                output_size_tgt, # the dimensionality of the output of y. In the paper it is always 1, but we can be more general.
                n_input_sensors, # the number of input sensors, "m" in the paper
                p=20, # This is the number of terms for the final dot product operation. In the paper, they say at least 10.
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

        # the conv layers
        layers_branch = []
        layers_branch.append(torch.nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1))
        layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers_branch.append(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
        layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers_branch.append(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.Flatten())
        # mlp layers
        layers_branch.append(torch.nn.Linear(3136, hidden_size))
        layers_branch.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            layers_branch.append(torch.nn.Linear(hidden_size, hidden_size))
            layers_branch.append(torch.nn.ReLU())
        layers_branch.append(torch.nn.Linear(hidden_size, output_size_tgt * p))
        self.model = torch.nn.Sequential(*layers_branch)

    def forward(self, x):
        ins = x.reshape(x.shape[0], 31, 31, x.shape[-1])
        ins = ins.permute(0, 3, 1, 2)
        outs = self.model(ins)
        return outs