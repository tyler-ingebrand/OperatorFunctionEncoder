from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import  TensorboardCallback

import argparse

from src.DeepONet import DeepONet
from src.Datasets.IntegralDataset import IntegralDataset
from src.OperatorEncoder import OperatorEncoder

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="inner_product")
parser.add_argument("--epochs", type=int, default=10_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
parser.add_argument("--model_type", type=str, default="FE")
args = parser.parse_args()
assert args.model_type in ["FE", "deeponet"]


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method if args.model_type == "FE" else args.model_type
seed = args.seed
load_path = args.load_path
residuals = args.residuals
model_type = args.model_type
if load_path is None:
    logdir = f"logs/integral_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

input_range=(-10, 10)
dataset = IntegralDataset(input_range=input_range, freeze_xs = args.model_type == "deeponet")

# create the model
if args.model_type == "FE":
    model = OperatorEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
else:
    model = DeepONet(input_size=dataset.input_size[0],
                     output_size=dataset.output_size[0],
                     n_input_sensors=dataset.n_examples_per_sample,
                     p=20,
                     ).to(device)
print("Architecture: ", args.model_type, f", Train method: {train_method}" if args.model_type == "FE" else "")
print("Model size: ", sum(p.numel() for p in model.parameters()))

# train or load
if load_path is None:
    # create callbacks
    callback = TensorboardCallback(logdir) # this one logs training data

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))





# plot
with torch.no_grad():
    n_plots = 9
    example_xs, example_ys, xs, transformed_ys, info = dataset.sample(device)

    # get predictions
    if args.model_type == "FE":
        y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=train_method)
    else:
        y_hats = model.forward(example_xs, example_ys, xs)  # deeponet

    # organize data for plotting
    xs, indicies = torch.sort(xs, dim=-2)
    transformed_ys = transformed_ys.gather(dim=-2, index=indicies)
    y_hats = y_hats.gather(dim=-2, index=indicies)


    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        true_values = info["As"][i].to(device) * xs[i] ** 2 + info["Bs"][i].to(device) * xs[i] + info["Cs"][i].to(device)

        # plots
        ax.plot(xs[i].cpu(), true_values.cpu(), label="Function")
        ax.plot(xs[i].cpu(), transformed_ys[i].cpu(), label="Integral")
        ax.plot(xs[i].cpu(), y_hats[i].cpu(), label="Estimated Integral")

        # labels
        if i == n_plots - 1:
            ax.legend()
        title = f"${info['As'][i].item():.2f}x^2 + {info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
        ax.set_title(title)

        # lims
        # y_min, y_max = transformed_ys[i].min().item(), transformed_ys[i].max().item()
        # ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()






    if args.model_type == "FE":
        # plot the basis functions
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)
        basis = model.model.forward(xs)
        for i in range(n_basis):
            ax.plot(xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
        if residuals:
            avg_function = model.average_function.forward(xs)
            ax.plot(xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

        plt.tight_layout()
        plt.savefig(f"{logdir}/basis.png")