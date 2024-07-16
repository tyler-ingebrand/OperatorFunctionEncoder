


from datetime import datetime
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import trange

from FunctionEncoder import MSECallback, ListCallback, TensorboardCallback
from src.OperatorDataset import QuadraticDataset, SinDataset, CombinedDataset
from src.SVDEncoder import SVDEncoder

def plot(dataset, dataset_type, model:SVDEncoder, device:str, train_method:str, logdir:str, input_range):
    with torch.no_grad():
        n_plots = 9

        # get data
        example_xs, example_ys, xs, ys, info = dataset.sample(device)



        # predict functions
        y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=train_method, dataset=dataset_type)

        # organize for plotting
        xs, indicies = torch.sort(xs, dim=-2)
        ys = ys.gather(dim=-2, index=indicies)
        y_hats = y_hats.gather(dim=-2, index=indicies)

        # plot
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        for i in range(n_plots):
            # maybe update ys if its the src dataset instead of target
            if dataset_type == "source":
                ys_i = info['As'][i].item() * xs[i] ** 2 + info['Bs'][i].item() * xs[i] + info['Cs'][i].item()
            else:
                ys_i = ys[i]

            ax = axs[i // 3, i % 3]
            ax.plot(xs[i].cpu(), ys_i.cpu(), label="True")
            ax.plot(xs[i].cpu(), y_hats[i].cpu(), label="Approximation")


            if i == n_plots - 1:
                ax.legend()
            if dataset_type == "source":
                title = f"${info['As'][i].item():.2f}x^2 + {info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
            elif dataset_type == "target":
                title = f"${info['As'][i].item():.2f}sin(x) + {info['Bs'][i].item():.2f}sin(4x) + {info['Cs'][i].item():.2f}sin(8x)"
            else:
                title = f"err"
            ax.set_title(title)
            y_min, y_max = ys[i].min().item(), ys[i].max().item()
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        name = dataset_type
        plt.savefig(f"{logdir}/{name}_plot.png")
        plt.clf()

        # plot the basis functions
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)

        if dataset_type == "source":
            basis = model.src_model.forward(xs)
        else:
            basis = model.tgt_model.forward(xs)

        for i in range(model.n_basis):
            ax.plot(xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
        if residuals:
            avg_function = model.average_function.forward(xs)
            ax.plot(xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

        plt.tight_layout()
        plt.savefig(f"{logdir}/{name}_basis.png")




# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/svd/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# first train the quadratic space encoder
dataset = CombinedDataset(calibration_only=False)

# create the model
model = SVDEncoder( input_size_src=(1,),
                        input_size_tgt=(1,),
                        output_size_src=(1,),
                        output_size_tgt=(1,),
                        data_type=dataset.data_type,
                        n_basis=n_basis,
                        method=train_method,
                        ).to(device)
if load_path is None:
    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = MSECallback(dataset, device=device, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
plot(dataset, "source", model, device, train_method, logdir, dataset.input_range_source)
plot(dataset, "target", model, device, train_method, logdir, dataset.input_range_target)


with torch.no_grad():
    # now plot 4 examples.
    # on the left, show quadratic functions
    # on the right, show sin functions
    # in the middle, show an arrow
    # make the first column width 1, the second width 0.2, the third width 1
    size = 5
    fig = plt.figure(figsize=(2.2 * size,  2.5 * size), dpi=300)
    gridspec = fig.add_gridspec(4, 3, width_ratios=[1, 0.4, 1])
    axs = gridspec.subplots()

    # get data and representations
    src_xs, src_ys, tgt_xs, tgt_ys, info = dataset.sample(device)
    rep, _ = model.compute_representation(src_xs, src_ys, method=train_method)

    for row in range(4):
        # plot the quadratic function
        representation = rep[row].unsqueeze(0)
        As, Bs, Cs = info["As"][row].item(), info["Bs"][row].item(), info["Cs"][row].item()

        # compute xs and true ys
        xs = torch.linspace(dataset.input_range_source[0], dataset.input_range_source[1], 1_000).reshape(1, 1_000, 1).to(device)
        ys = As * xs ** 2 + Bs * xs + Cs

        # compute y_hats
        y_hat = model.predict(xs, representation, dataset="source")

        # plot
        ax = axs[row, 0]
        ax.plot(xs.flatten().cpu(), ys.flatten().cpu(), label="$f(x)$")
        ax.plot(xs.flatten().cpu(), y_hat.flatten().cpu(), label="$\hat{f}(x)$")
        ax.set_title(f"${As:.2f}x^2 + {Bs:.2f}x + {Cs:.2f}$")
        if row == 3:
            ax.legend()


        # add an arrow to the middle column
        # and a T right above it
        ax = axs[row, 1]
        ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        ax.text(0.1, 0.1, "T", fontsize=30)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.axis("off")


        # plot the sin function, based on the estimated transformed representation
        ys = As * torch.sin(xs) + Bs * torch.sin(4 * xs) + Cs * torch.sin(8 * xs)

        # compute y_hats
        y_hat = model.predict(xs, representation, dataset="target")

        # plot
        ax = axs[row, 2]
        ax.plot(xs.flatten().cpu(), ys.flatten().cpu(), label="$Tf(x)$")
        ax.plot(xs.flatten().cpu(), y_hat.flatten().cpu(), label="$\hat{Tf}(x)$")
        ax.set_title(f"${As:.2f}sin(x) + {Bs:.2f}sin(4x) + {Cs:.2f}sin(8x)$")
        if row == 3:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{logdir}/transformation_plot.png")


