import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_specs import colors, labels, titles

plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

# params
exp_dir = "logs_ablation_sensor"
datasets = ["Integral", "Derivative"]
csv_name = "plot.csv"
algs = ["SVD_least_squares", "matrix_least_squares", "Eigen_least_squares","deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]


for dataset in datasets:
    logdir = f"{exp_dir}/{dataset}/"
    
    # load the csv file
    data_matrix = np.loadtxt(os.path.join(logdir, csv_name), delimiter=",", skiprows=1)

    # x same for all algs
    xs = data_matrix[:, 0]

    # create plot
    fig, ax = plt.subplots()

    for i, alg in enumerate(algs):
        # get the y values
        ys_median = data_matrix[:, 3*i+1]
        if ys_median[0] == 0: # some algs dont run on a dataset, so we skip them.
            continue
        ys_q1 = data_matrix[:, 3*i+2]
        ys_q3 = data_matrix[:, 3*i+3]

        # some algs completely fail on some datasets. This happens if there q1 is greater than the upper bound.
        # in this case, print an error message and skip the alg.
        # if np.min(ys_q1) > upper_bounds[dataset]:
        #     print(f"{alg} failed on {dataset}, with a median terminal score of {np.mean(ys_median[-30:])}")
        #     continue

        # plot the median
        color = colors[alg]
        label = labels[alg]
        ax.plot(xs, ys_median, label=label, color=color,)
        ax.fill_between(xs, ys_q1, ys_q3, alpha=0.3, color=color, lw=0.0)

    # set the y limits
    # ax.set_ylim(-0.1 if dataset in ["Derivative", "Integral"] else 0.0, upper_bounds[dataset])
    # ax.set_xlim(0, 1000)

    # set log y scaling
    ax.set_yscale("log")

    # set the title
    title = titles[dataset]
    plt.title(title)

    # set the labels
    ax.set_xlabel("Number of Input Sensors")
    ax.set_ylabel("Test MSE after 70k Steps")

    # set the tick labels
    ax.set_xticks([10, 50, 100, 200, 400, 600, 800, 1000])

    # set the legend
    ax.legend(frameon=False, ncol=2)

    # save the plot
    plot_name = f"plot_ablation_sensor_{dataset}.pdf"
    plt.savefig(os.path.join(logdir, plot_name))