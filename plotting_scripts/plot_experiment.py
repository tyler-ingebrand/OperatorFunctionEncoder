import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


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

datasts = ["Integral", "Derivative", "Elastic", "Darcy", "Heat", "LShaped"]
algs = ["SVD_least_squares", "matrix_least_squares", "Eigen_least_squares","deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]
logdir = "logs_experiment"

# for every dataset type
table = {}
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

            data[alg] = {"raw_data_values": [], "raw_data_steps": []}
            # list all subdirectories. Each is a seed trial run
            subdirs = [f.path for f in os.scandir(log_alg_dir) if f.is_dir()]

            # iterate subdirs and add to list
            for subdir in subdirs:
                print("\t\t", subdir)

                # read the tensorboard data
                d = read_tensorboard(subdir, ["test/mse"])
                data[alg]["raw_data_steps"].append(d["test/mse"][:, 0])
                data[alg]["raw_data_values"].append(d["test/mse"][:, 1])
            
            # compute median and quartiles
            raw_data = np.stack(data[alg]["raw_data_values"], axis=0)
            quarts = np.percentile(raw_data, [25, 50, 75], axis=0)
            data[alg]["median"] = quarts[1]
            data[alg]["q1"] = quarts[0]
            data[alg]["q3"] = quarts[2]
            data[alg]["mean_final_value"] = np.mean(raw_data[:, -1])
            data[alg]["std_final_value"] = np.std(raw_data[:, -1])


            # plot with fill between
            ax.plot(data[alg]["raw_data_steps"][0], data[alg]["median"], label=alg)
            ax.fill_between(data[alg]["raw_data_steps"][0], data[alg]["q1"], data[alg]["q3"], alpha=0.3)
        except Exception as e:
            print(e)
            continue
    miny = 0
    maxy = max(data["deeponet"]["q3"][15], data["matrix_least_squares"]["q3"][15])
    ax.set_ylim(miny, maxy)
    plt.title(dataset)
    ax.legend()
    plt.savefig(os.path.join(log_dataset_dir, "plot.png"))

    # also save to csv for plotting in latex later.
    # create headers
    col_headers = ["step"]
    for alg in algs:
        col_headers.append(f"{alg}_median")
        col_headers.append(f"{alg}_q1")
        col_headers.append(f"{alg}_q3")

    # fetch data
    data_matrix = np.zeros((len(data["matrix_least_squares"]["raw_data_steps"][0]), 3*len(algs)+1))
    data_matrix[:, 0] = data["matrix_least_squares"]["raw_data_steps"][0]
    for i, alg in enumerate(algs):
        try:
            data_matrix[:, 3*i+1] = data[alg]["median"]
            data_matrix[:, 3*i+2] = data[alg]["q1"]
            data_matrix[:, 3*i+3] = data[alg]["q3"]
        except:
            continue

    # save to csv
    np.savetxt(os.path.join(log_dataset_dir, "plot.csv"), data_matrix, delimiter=",", header=",".join(col_headers), comments="")

    # add mean and std to table
    table[dataset] = {}
    for alg in algs:
        if alg in data:
            table[dataset][alg] = (data[alg]["mean_final_value"], data[alg]["std_final_value"])

# print the mean and std of the final values for all datasts for each alg
# $1.31\mathrm{e}{0} \pm 1.04\mathrm{e}{0}$
# do this in latex format so i can copy paste into document
for d in datasts:
    print("\n")
    print(d, end="")
    print(" & ", end="")
    for alg in algs:
        # skip these
        if "cnn" in alg:
            continue


        try:
            # get the two parts of the scientific notation
            mean, std = table[d][alg]
            mean_str = f"{mean:0.2E}".split("E")
            std_str = f"{std:0.2E}".split("E")
            print(f"${mean_str[0]}\\mathrm{{e}}{{{mean_str[1]}}} \\pm {std_str[0]}\\mathrm{{e}}{{{std_str[1]}}}$", end="")
        except:
            print(" - ", end="")
        print(" & ", end="")
print()