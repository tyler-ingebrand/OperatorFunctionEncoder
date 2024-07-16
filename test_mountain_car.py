


from datetime import datetime
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import trange

from FunctionEncoder import FunctionEncoder, TensorboardCallback
from src.Datasets.MountainCarPoliciesDataset import MountainCarPoliciesDataset, MountainCarEpisodesDataset, \
    MountainCarCombinedDataset

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis_src", type=int, default=11)
parser.add_argument("--n_basis_tgt", type=int, default=12)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis_src = args.n_basis_src
n_basis_tgt = args.n_basis_tgt
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/mountain_car/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# first train the quadratic space encoder
policy_dataset = MountainCarPoliciesDataset()

# create the model
policy = FunctionEncoder(input_size=policy_dataset.input_size,
                        output_size=policy_dataset.output_size,
                        data_type=policy_dataset.data_type,
                        n_basis=n_basis_src,
                        method=train_method,
                        use_residuals_method=residuals).to(device)
if load_path is None:
    # create callbacks
    callback = TensorboardCallback(logdir) # this one logs training data

    # train the model
    policy.train_model(policy_dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(policy.state_dict(), f"{logdir}/policy.pth")
else:
    # load the model
    policy.load_state_dict(torch.load(f"{logdir}/policy.pth"))


# next do the target function space, sin
trajectory_dataset = MountainCarEpisodesDataset()

# create the model
trajectory_model = FunctionEncoder(input_size=trajectory_dataset.input_size,
                        output_size=trajectory_dataset.output_size,
                        data_type=trajectory_dataset.data_type,
                        n_basis=n_basis_tgt,
                        method=train_method,
                        use_residuals_method=residuals).to(device)

if load_path is None:
    # train the model
    trajectory_model.train_model(trajectory_dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(trajectory_model.state_dict(), f"{logdir}/trajectory_model.pth")
else:
    # load the model
    trajectory_model.load_state_dict(torch.load(f"{logdir}/trajectory_model.pth"))


# now try to convert between them
combined_dataset = MountainCarCombinedDataset(policy_dataset, trajectory_dataset)
with torch.no_grad():
    all_src_Cs = []
    all_tgt_Cs = []
    # collect a bunch of data. We have to accumulate to save memory
    for epoch in trange(100):
        src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device)

        # then get the representations
        src_Cs, _ = policy.compute_representation(src_xs, src_ys, method=train_method)
        tgt_Cs, _ = trajectory_model.compute_representation(tgt_xs, tgt_ys, method=train_method)

        all_src_Cs.append(src_Cs)
        all_tgt_Cs.append(tgt_Cs)

    src_Cs = torch.cat(all_src_Cs, dim=0)
    tgt_Cs = torch.cat(all_tgt_Cs, dim=0)

    # now compute the transformation via LS solution.
    A = torch.linalg.lstsq(src_Cs, tgt_Cs).solution.T
    print("Transformation error: ", torch.mean(torch.norm(tgt_Cs - src_Cs @ A.T, dim=-1)).item())

# first plot the policy space
with torch.no_grad():
    example_xs, example_ys, _, _, info = policy_dataset.sample(device)
    # create a grid of xs
    x = torch.linspace(-1.2, 0.6, 100)
    y = torch.linspace(-0.07, 0.07, 100)
    xx, yy = torch.meshgrid(x, y)
    xs = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)

    # groundtruth ys
    As, Bs, Cs, Ds, Es, Fs = info["As"], info["Bs"], info["Cs"], info["Ds"], info["Es"], info["Fs"]
    As, Bs, Cs, Ds, Es, Fs = As.unsqueeze(1), Bs.unsqueeze(1), Cs.unsqueeze(1), Ds.unsqueeze(1), Es.unsqueeze(1), Fs.unsqueeze(1)
    As, Bs, Cs, Ds, Es, Fs = As.to(device), Bs.to(device), Cs.to(device), Ds.to(device), Es.to(device), Fs.to(device)
    ys = As * xs[:, 0:1].unsqueeze(0) ** 2 + \
        Bs * xs[:, 0:1].unsqueeze(0) * xs[:, 1:2].unsqueeze(0) + \
        Cs * xs[:, 1:2].unsqueeze(0) ** 2 + \
        Ds * xs[:, 0:1].unsqueeze(0) + \
        Es * xs[:, 1:2].unsqueeze(0) + \
        Fs

    # predict policy outputs
    y_hats = policy.predict_from_examples(example_xs, example_ys, xs.reshape(1, -1, 2).repeat(example_xs.shape[0], 1, 1), method=train_method)
    v_max = max(torch.max(ys), torch.max(y_hats))
    v_min = min(torch.min(ys), torch.min(y_hats))

    # now plot comparisons. We plot the groundtruth on the left and the predicted on the right, 2 cols, 4 rows
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    for i in range(4):
        ax = axs[i, 0]
        ax.set_title(f"Groundtruth Policy {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.scatter(xs[:, 0].cpu(), xs[:, 1].cpu(), c=ys[i].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)

        ax = axs[i, 1]
        ax.set_title(f"Predicted Policy {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.scatter(xs[:, 0].cpu(), xs[:, 1].cpu(), c=y_hats[i].cpu(), cmap="coolwarm", vmin=v_min, vmax=v_max)


    plt.savefig(f"{logdir}/policy.png")

# now plot the predicted and true trajectories
with torch.no_grad():
    example_xs, example_ys, xs, ys, info = trajectory_dataset.sample(device)

    # predict trajectory outputs
    y_hats = trajectory_model.predict_from_examples(example_xs, example_ys, xs, method=train_method)

    # now plot comparisons. We plot the groundtruth on the left and the predicted on the right, 2 cols, 4 rows
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    cmap = plt.get_cmap("rainbow")

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

    plt.savefig(f"{logdir}/trajectory.png")


with torch.no_grad():
    # now plot 4 examples.
    # on the left, show policies
    # on the right, show trajectories
    # in the middle, show an arrow
    # make the first column width 1, the second width 0.2, the third width 1
    size = 5
    fig = plt.figure(figsize=(2.2 * size,  2.5 * size), dpi=300)
    gridspec = fig.add_gridspec(4, 3, width_ratios=[1, 0.4, 1])
    axs = gridspec.subplots()

    # get data and representations
    src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device)
    src_Cs, _ = policy.compute_representation(src_xs, src_ys, method=train_method)

    # create a grid of xs
    x = torch.linspace(-1.2, 0.6, 100)
    y = torch.linspace(-0.07, 0.07, 100)
    xx, yy = torch.meshgrid(x, y)
    xs = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)


    for row in range(4):
        # plot the quadratic function
        representation = src_Cs[row].unsqueeze(0)
        As, Bs, Cs, Ds, Es, Fs = info["As"][row], info["Bs"][row], info["Cs"][row], info["Ds"][row], info["Es"][row], info["Fs"][row]

        # compute xs and true ys for the src space
        ys = As * xs[:, 0:1].unsqueeze(0) ** 2 + \
             Bs * xs[:, 0:1].unsqueeze(0) * xs[:, 1:2].unsqueeze(0) + \
             Cs * xs[:, 1:2].unsqueeze(0) ** 2 + \
             Ds * xs[:, 0:1].unsqueeze(0) + \
             Es * xs[:, 1:2].unsqueeze(0) + \
             Fs

        # compute y_hats
        y_hat = trajectory_model.predict(xs, representation)

        # plot
        ax = axs[row, 0]
        ax.set_title(f"Predicted Policy {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.scatter(xs[:, 0].cpu(), xs[:, 1].cpu(), c=y_hat.cpu(), cmap="coolwarm")

        # add an arrow to the middle column
        # and a T right above it
        ax = axs[row, 1]
        ax.arrow(0, 0, 0.25, 0.0, head_width=0.1, head_length=0.1, fc='black', ec='black', lw=15)
        ax.text(0.1, 0.1, "T", fontsize=30)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.axis("off")


        # plot the sin function, based on the estimated transformed representation
        representation = src_Cs[row:row+1] @ A.T
        ys = combined_dataset.trajectory_dataset.trajectory(info["As"], info["Bs"], info["Cs"], info["Ds"], info["Es"],info["Fs"])
        xs = torch.arange(0, combined_dataset.trajectory_dataset.max_time).unsqueeze(0).unsqueeze(2).expand(combined_dataset.n_functions_per_sample, combined_dataset.trajectory_dataset.max_time, 1).to(torch.float32).to(device)

        # compute y_hats
        y_hat = trajectory_model.predict(xs, representation)

        # plot
        ax = axs[row, 2]
        ax.set_title(f"Predicted Trajectory {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.plot(y_hat[row][:, 0].cpu(), y_hat[row][:, 1].cpu(), "Estimated")
        ax.plot(ys[row][:, 0].cpu(), ys[row][:, 1].cpu(), "Groundtruth")
        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-0.07, 0.07)



    plt.tight_layout()
    plt.savefig(f"{logdir}/transformation_plot.png")







