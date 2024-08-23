import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import ConnectionPatch, PathPatch
from matplotlib.path import Path
from plotting_specs import colors, labels, titles

plt.rcParams.update({'font.size': 20})
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

def data_to_normalized(point, ax, width, height, dpi):
    point = ax.transData.transform(point)
    return (point[0] / (width * dpi), point[1] / (height * dpi))


# params
exp_dir = "logs_experiment"
datasets = ["Derivative", "Integral"]
csv_name = "plot.csv"
algs = ["SVD_least_squares", "matrix_least_squares", "Eigen_least_squares","deeponet", "deeponet_cnn", "deeponet_pod", "deeponet_2stage", "deeponet_2stage_cnn"]

upper_bounds = {
    "Derivative": {False: 175, True: 1},
    "Integral": {False: 315, True: 1},
}

# make a plot with 2 rows, 5 columns
#width = 16.5cm
wspace = 0.5
hspace = 0.3
width = 20 + 2 * wspace
height = 8 + hspace
dpi = 100
fig = plt.figure(constrained_layout=True, figsize=(width, height), dpi=dpi)
gs = fig.add_gridspec(2,5, wspace=wspace, hspace=hspace)

axs = { "Integral": {True: fig.add_subplot(gs[0, 2]),
                     False: fig.add_subplot(gs[0:2, 0:2])},
        "Derivative": {True: fig.add_subplot(gs[1, 2]), 
                       False: fig.add_subplot(gs[0:2, 3:5])}
}

for dataset in datasets:
    for linear_focus in [False, True]:
        ax = axs[dataset][linear_focus]
        logdir = f"{exp_dir}/{dataset}/"
        
        # load the csv file
        data_matrix = np.loadtxt(os.path.join(logdir, csv_name), delimiter=",", skiprows=1)

        # x same for all algs
        xs = data_matrix[:, 0]


        for i, alg in enumerate(algs):
            # get the y values
            ys_median = data_matrix[:, 3*i+1]
            if ys_median[0] == 0: # some algs dont run on a dataset, so we skip them.
                continue
            ys_q1 = data_matrix[:, 3*i+2]
            ys_q3 = data_matrix[:, 3*i+3]

            # some algs completely fail on some datasets. This happens if there q1 is greater than the upper bound.
            # in this case, print an error message and skip the alg.
            if np.min(ys_q1) > upper_bounds[dataset][linear_focus]:
                print(f"{alg} failed on {dataset}, with a median terminal score of {np.mean(ys_median[-30:])}")
                continue

            # plot the median
            color = colors[alg]
            label = labels[alg]
            ax.plot(xs, ys_median, label=label, color=color,)
            ax.fill_between(xs, ys_q1, ys_q3, alpha=0.3, color=color, lw=0.0)

        # set the y limits
        ax.set_ylim(-0.1 if not linear_focus else 0.0, upper_bounds[dataset][linear_focus])
        ax.set_xlim(-1000, 30_000)

        # set the title
        title = titles[dataset]
        ax.set_title(title)

        # set the labels
        if dataset == "Derivative" and linear_focus:
            ax.set_xlabel("Gradient Steps")
        if dataset == "Integral" and not linear_focus:
            ax.set_ylabel("Test MSE")

        # if derivative dataset, plot the y ticks on the right axis instead of left
        if (dataset == "Integral" and linear_focus):
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")


        # set the tick labels
        ax.set_xticks([0, 10000, 20000, 30000])
        if linear_focus:
            if dataset == "Derivative":
                ax.set_yticks([0, 0.33, 0.66, 1])
            else:
                ax.set_yticks([0.33, 0.66, 1])
        else:
            if dataset == "Derivative":
                ax.set_yticks([ 50, 100, 150])
            else:
                ax.set_yticks([0, 100, 200, 300])

        # # set the legend
        # if dataset == "Derivative" and not linear_focus:
        #     ax.legend()

# legend at bottom of figure
# make the lines in the legend thicker
legend = axs["Integral"][False].legend(loc='upper center', bbox_to_anchor=(1.45, -0.07), ncol=8, frameon=False, fontsize=20)
for legend_handle in legend.legend_handles:
    legend_handle.set_linewidth(4.0)

# post processing, draw lines connecting the left figure to the upper,center figure
# and the right figure to the lower,center figure.
left_ax = axs["Integral"][False]
upper_center = axs["Integral"][True]
right_ax = axs["Derivative"][False]
lower_center = axs["Derivative"][True]

# set tight layout with 0.1 vspace and hspace
plt.tight_layout(h_pad=0.5, w_pad=2.0, rect=[0.2, 0, 0.8, 1])
plt.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.13)
# upper left
xy1 = (30_000, 1.0)
xy2 = (-1000, 1.0)
xy1, xy2 = data_to_normalized(xy1, left_ax, width, height, dpi), data_to_normalized(xy2, upper_center, width, height, dpi)
middle_x = xy1[0] * 0.8 + xy2[0] * 0.2
middle_y = xy1[1] * 0.8 + xy2[1] * 0.2

points = [xy1, (middle_x, xy1[1]), (middle_x, middle_y), (middle_x, xy2[1]), xy2]
path = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
path2 = Path(points, path)
patch = PathPatch(path2, lw=1.0, color='black', fill=False, linestyle='dashed')
fig.add_artist(patch)


# # # lower left
xy1 = (30_000, -0.1)
xy2 = (-1000, 0)
xy1, xy2 = data_to_normalized(xy1, left_ax, width, height, dpi), data_to_normalized(xy2, upper_center, width, height, dpi)
middle_x = xy1[0] * 0.7 + xy2[0] * 0.3
middle_y = xy1[1] * 0.7 + xy2[1] * 0.3

points = [xy1, (middle_x, xy1[1]), (middle_x, middle_y), (middle_x, xy2[1]), xy2]
path = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
path2 = Path(points, path)
patch = PathPatch(path2, lw=1.0, color='black', fill=False, linestyle='dashed')
fig.add_artist(patch)


# # upper right
xy1 = (-1000, upper_bounds["Derivative"][True])
xy2 = (30_000, upper_bounds["Derivative"][True])
xy1, xy2 = data_to_normalized(xy1, right_ax, width, height, dpi), data_to_normalized(xy2, lower_center, width, height, dpi)
middle_x = xy1[0] * 0.2 + xy2[0] * 0.8
middle_y = xy1[1] * 0.2 + xy2[1] * 0.8

points = [xy1, (middle_x, xy1[1]), (middle_x, middle_y), (middle_x, xy2[1]), xy2]
path = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
path2 = Path(points, path)
patch = PathPatch(path2, lw=1.0, color='black', fill=False, linestyle='dashed')
fig.add_artist(patch)

# lower right
xy1 = (-1000, -0.1)
xy2 = (30_000, 0)
xy1, xy2 = data_to_normalized(xy1, right_ax, width, height, dpi), data_to_normalized(xy2, lower_center, width, height, dpi)
middle_x = xy1[0] * 0.8 + xy2[0] * 0.2
middle_y = xy1[1] * 0.8 + xy2[1] * 0.2 + 0.000

points = [xy1, (middle_x, middle_y), xy2]
path = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
path2 = Path(points, path)
patch = PathPatch(path2, lw=1.0, color='black', fill=False, linestyle='dashed')
fig.add_artist(patch)

# save the plot
plot_name = f"linear.pdf"
plt.savefig(os.path.join(exp_dir, plot_name))