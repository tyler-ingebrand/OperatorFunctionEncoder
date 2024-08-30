from FunctionEncoder import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.DeepONet import DeepONet
from tqdm import tqdm
import matplotlib.ticker as ticker

class GradPathCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.parameters = []

    def on_training_start(self, locals: dict) -> None:
        with torch.no_grad():
            if len(self.parameters) == 0:
                model = locals['self']
                params = [p.clone() for p in model.parameters()]
                self.parameters.append(params)

    def on_step(self, locals: dict) -> None:
        with torch.no_grad():
            model = locals['self']
            params = [p.clone() for p in model.parameters()]
            self.parameters.append(params)


    def loss_fe(self, model, example_xs, example_ys, xs, ys):
        # approximate functions, compute error
        representation, gram = model.compute_representation(example_xs, example_ys, method=model.method)
        y_hats = model.predict(xs, representation, precomputed_average_ys=None)
        prediction_loss = model._distance(y_hats, ys, squared=True).mean()

        # LS requires regularization since it does not care about the scale of basis
        # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
        # from going to infinity.
        if model.method == "least_squares":
            norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()

        # add loss components
        loss = prediction_loss
        if model.method == "least_squares":
            loss = loss + model.regularization_parameter * norm_loss

        return loss

    def loss_deeponet(self, model, example_xs, example_ys, xs, ys):
        # sample input data
        xs, u_xs, ys, G_u_ys = example_xs, example_ys, xs, ys

        # approximate functions, compute error
        estimated_G_u_ys = model.forward(xs, u_xs, ys)
        prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)
        return prediction_loss

    def unflatten_grads(self, grads, model):
        unflattened_grads = []
        start = 0
        for p in model.parameters():
            size = p.numel()
            unflattened_grads.append(grads[start:start+size].reshape(*p.shape))
            start += size
        return unflattened_grads

    def plot_grad_path(self, model, dataset, savepath, density=20):
        with torch.no_grad():
            # get initial parameters 
            init_params = self.parameters[0]
            
            
            # first compute the difference between every two consecutive parameters
            grads = []
            for i in range(len(self.parameters) - 1):
                grads.append([p1 - p0 for p0, p1 in zip(self.parameters[i], self.parameters[i + 1])])
            
            # next, find the priniciple components of the gradients
            flattened_grads = [torch.concat([p.flatten() for p in step]).flatten() for step in grads]
            flattened_grads = torch.stack(flattened_grads)

            # compute the mean and recenter flattened_grads
            mean = torch.mean(flattened_grads, dim=0)
            centered_grads = flattened_grads - mean

            # do svd to get the 2 principle components
            U, S, V = torch.svd(centered_grads)
            principle_components = V[:, :2].T

            # project the centered_grads onto the principle components to get 2d coordinate representation
            projection = torch.matmul(principle_components, centered_grads.T).T

            # make a grid of points between start and end
            # with some buffer around each of them, as a percent
            buffer = 0.2
            max_dim0 = torch.max(projection[:, 0]).item()
            max_dim1 = torch.max(projection[:, 1]).item()
            min_dim0 = torch.min(projection[:, 0]).item()
            min_dim1 = torch.min(projection[:, 1]).item()
            x = torch.linspace(min_dim0 - buffer * (max_dim0 - min_dim0), max_dim0 + buffer * (max_dim0 - min_dim0), density)
            y = torch.linspace(min_dim1 - buffer * (max_dim1 - min_dim1), max_dim1 + buffer * (max_dim1 - min_dim1), density)

            # make a meshgrid
            X, Y = torch.meshgrid(x, y, indexing='ij')

            # get data from dataset
            example_xs, example_ys, xs, ys, _ = dataset.sample(None)

            # for each of these points, compute the loss to make a contour plot
            Z = torch.zeros_like(X)
            tbar = tqdm(total=density**2)
            for i in range(density):
                for j in range(density):
                    # compute the current model parameters
                    point = torch.tensor([X[i, j], Y[i, j]], device=principle_components.device)
                    normalized_grads = torch.matmul(principle_components.T, point)
                    grads = normalized_grads + mean
                    unflattened_grads = self.unflatten_grads(grads, model)
                    model_params = [p - g for p, g in zip(init_params, unflattened_grads)]

                    # load them into model
                    model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), model_params)})
                    
                    # compute the loss
                    if isinstance(model, DeepONet):
                        loss = self.loss_deeponet(model, example_xs, example_ys, xs, ys)
                    else:
                        loss = self.loss_fe(model, example_xs, example_ys, xs, ys)
                    Z[i, j] = loss.item()
                    tbar.update(1)
            
            # do log of Z because there is such a large variation
            Z = torch.log(Z + 1e-10)

            # plot the gradient path
            fig, ax = plt.subplots(figsize=(8, 6.5), dpi=500)
            ax.scatter(projection[0, 0].cpu(), projection[0, 1].cpu(), c='r', label='Initial Parameters')
            ax.plot(projection[:, 0].cpu(), projection[:, 1].cpu(), 'black', label='Gradient Path')

            # do the contour plot
            c = ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=20)

            # format the labels
            fmt = lambda x: f'{np.exp(x):.2e}'

            #  add labels
            ax.clabel(c, c.levels[3::4], inline=True, fmt=fmt, fontsize=10)

            # add a legend outside the figure
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

            # lables
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('Gradient Path')
            plt.tight_layout()
            plt.savefig(savepath)
            del fig, ax