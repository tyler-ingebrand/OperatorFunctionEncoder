import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import torch

def read_tensorboard(logdir, scalars):
    """returns a dictionary of numpy arrays for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    for s in scalars:
        assert s in ea.Tags()["scalars"], f"{s} not found in event accumulator"
    data = {k: np.array([[e.step, e.value] for e in ea.Scalars(k)]) for k in scalars}
    return data

datasts = [ "Integral", "Derivative"]
algs = ["SVD_least_squares", "matrix_least_squares", "Eigen_least_squares","deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]
n_sensors = (10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
logdir = "logs_ablation_sensor"

# for every dataset type
for dataset in datasts:
    print(dataset)
    log_dataset_dir = os.path.join(logdir, dataset)
    data = {}

    # create plots for each dataset, just to visualize
    fig, ax = plt.subplots()


    # for every algorithm
    for alg in algs:
        try:
            print("\t", alg)
            log_alg_dir = os.path.join(log_dataset_dir, alg)
            
            # if the log alg dir doesnt exist, the alg isnt applicable to this dataset
            # so just continue
            if not os.path.exists(log_alg_dir):
                print("\t\tN/A")
                continue

            data[alg] = {n : {"raw_data_values": [], "raw_data_steps": []} for n in n_sensors}
            # list all subdirectories. Each is a seed trial run
            subdirs = [f.path for f in os.scandir(log_alg_dir) if f.is_dir()]

            # iterate subdirs and add to list
            for subdir in subdirs:
                print("\t\t", subdir)

                params = torch.load(os.path.join(subdir, "params.pth"), weights_only=True)
                sens = params["n_sensors"]

                # read the tensorboard data
                d = read_tensorboard(subdir, ["test/mse"])
                data[alg][sens]["raw_data_steps"].append(d["test/mse"][:, 0])
                data[alg][sens]["raw_data_values"].append(d["test/mse"][:, 1])
            
            # compute median and quartiles
            for ns in n_sensors:
                raw_data = np.stack(data[alg][ns]["raw_data_values"], axis=0)
                quarts = np.percentile(raw_data, [0, 50, 100], axis=0)
                data[alg][ns]["median"] = quarts[1]
                data[alg][ns]["q1"] = quarts[0]
                data[alg][ns]["q3"] = quarts[2]

            # Create data with n_Basis on x and final median/quartiles on y
            xs, ys, q1, q3 = [], [], [], []
            for ns in n_sensors:
                xs.append(ns)
                ys.append(data[alg][ns]["median"][-10:].mean())
                q1.append(data[alg][ns]["q1"][-10:].mean())
                q3.append(data[alg][ns]["q3"][-10:].mean())
            
            # add data to data dict
            data[alg]["xs"] = xs
            data[alg]["ys"] = ys
            data[alg]["q1"] = q1
            data[alg]["q3"] = q3


            # plot with fill between
            ax.plot(xs, ys, label=alg)
            ax.fill_between(xs, q1, q3, alpha=0.3)
        except Exception as e:
            print(e)
            continue
    # miny = 0
    # maxy = max(data["deeponet"]["q3"])
    # ax.set_ylim(miny, maxy)
    plt.xlabel("Number of Sensors")
    plt.ylabel("MSE after 70k Steps")
    plt.title(dataset)
    ax.legend()
    print("Saving to ", os.path.join(log_dataset_dir, "plot.png"))
    plt.savefig(os.path.join(log_dataset_dir, "plot.png"))

    # also save to csv for plotting in latex later.
    # create headers
    col_headers = ["n_sensors"]
    for alg in algs:
        col_headers.append(f"{alg}_median")
        col_headers.append(f"{alg}_q1")
        col_headers.append(f"{alg}_q3")

    # fetch data
    data_matrix = np.zeros((len(n_sensors), 3*len(algs)+1))
    data_matrix[:, 0] = data["deeponet"]["xs"]
    for i, alg in enumerate(algs):
        try:
            data_matrix[:, 3*i+1] = data[alg]["ys"]
            data_matrix[:, 3*i+2] = data[alg]["q1"]
            data_matrix[:, 3*i+3] = data[alg]["q3"]
        except:
            continue

    # save to csv
    np.savetxt(os.path.join(log_dataset_dir, "plot.csv"), data_matrix, delimiter=",", header=",".join(col_headers), comments="")

