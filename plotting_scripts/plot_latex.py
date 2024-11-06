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
linear_focus = False
exp_dir = "logs_experiment"
datasets = ["Derivative", "Integral", "Elastic", "LShaped", "Heat", "Darcy", "Burger"]
csv_name = "plot.csv"
algs = ["SVD_least_squares", "matrix_least_squares", "Eigen_least_squares","deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]


upper_bounds = {
    "Derivative": 0.5 if not linear_focus else 1,
    "Elastic": 0.006,
    "Integral": 1.0 if not linear_focus else 2,
    "LShaped": 0.08,
    "Heat": 0.0055,
    "Darcy": 0.0029,
}

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
    # ax.set_ylim(0.0, upper_bounds[dataset])
    x_max = xs.max()
    ax.set_xlim(0, x_max)

    # if derivatve or integral, set to logy scale
    # if dataset in ["Derivative", "Integral"]:
    ax.set_yscale("log")

    # if integral example, set yticks to 10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6
    if dataset == "Integral":
        # ax.set_yticks([10**i for i in range(-3, 7)])
        
        # # enable the minor ticks
        # ax.yaxis.set_minor_locator(plt.FixedLocator([10**i for i in range(-3, 7)]))
        # ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
        # ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))
        ax.set_ylim(None, 10**4)

    
    
    # set the title
    title = titles[dataset]
    plt.title(title)

    # set the labels
    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("Test MSE")

    # set the tick labels
    ax.set_xticks([0, 17500, 35000, 52500, 70000])

    # set the legend, no border
    if dataset not in ["Integral", "Darcy", "Heat"]:
        if dataset == "Burger":
            # outer right legend
            leg = ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.43, 1.0))
        else:
            leg = ax.legend(frameon=False, ncol=2)
        for line in leg.get_lines():
            line.set_linewidth(3.0)

    # save the plot
    plot_name = f"plot_{dataset}{'_fe_focus' if linear_focus and dataset in ['Derivative', 'Integral'] else ''}.pdf"
    plt.savefig(os.path.join(logdir, plot_name)) # , bbox_inches="tight")