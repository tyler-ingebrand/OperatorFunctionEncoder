from datetime import datetime
import os
from tqdm import trange
import matplotlib.pyplot as plt
import torch

# the base FE algorithm
from FunctionEncoder import TensorboardCallback, FunctionEncoder

# code used to compute matrix
from src.MatrixMethodHelpers import compute_A, train_nonlinear_transformation, get_num_parameters, get_num_layers, predict_number_params, get_hidden_layer_size, check_parameters

# import datasets
from src.Datasets.DerivativeDataset import CubicDerivativeDataset, CubicDataset, plot_source_cubic, plot_target_cubic_derivative, plot_transformation_derivative
from src.Datasets.OperatorDataset import CombinedDataset


# hyper params
epochs = 1_000
n_basis = 100
n_sensors = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
model_type = "matrix"
dataset_type = "Derivative"
transformation_type = "linear"
n_layers = 4
hidden_size = 128
freeze_example_xs = True # or False, B2B can do either one.

# generate logdir
logdir = f"logs_mwe/Derivative/matrix_least_squares/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# seed torch
torch.manual_seed(seed)

# generate datasets
src_dataset = CubicDataset(freeze_example_xs=freeze_example_xs, n_examples_per_sample=n_sensors, device=device)
tgt_dataset = CubicDerivativeDataset(n_examples_per_sample=n_sensors, freeze_xs=False, device=device)
combined_dataset = CombinedDataset(src_dataset, tgt_dataset, calibration_only=True) # calibration = True because we only use this dataset to compute the transformation matrix.

##############   Initialize    #############################################################################################################
src_model = FunctionEncoder(input_size=src_dataset.input_size,
                            output_size=src_dataset.output_size,
                            data_type=src_dataset.data_type,
                            n_basis=n_basis,
                            method="least_squares",
                            model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                            ).to(device)
tgt_model = FunctionEncoder(input_size=tgt_dataset.input_size,
                            output_size=tgt_dataset.output_size,
                            data_type=tgt_dataset.data_type,
                            n_basis=n_basis+1, # note this makes debugging way easier.
                            method="least_squares",
                            model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size},
                            ).to(device)

##############   Train    ##################################################################################################################

# Train src function encoder
print("Training source function encoder to span the space of Cubic Functions")
callback = TensorboardCallback(logdir=logdir, prefix="source")
src_model.train_model(src_dataset, epochs=epochs, callback=callback, progress_bar=True)

# Train the tgt function encoder
print("Training target function encoder to span the space of Quadratic Functions")
callback2 = TensorboardCallback(tensorboard=callback.tensorboard, prefix="target") # this logs to the same tensorboard but with a different prefix
tgt_model.train_model(tgt_dataset, epochs=epochs, callback=callback2, progress_bar=True)

# compute the transformation matrix between them
print("Computing the transformation matrix")
transformation = compute_A(src_model, tgt_model, combined_dataset, device, "least_squares", callback)
    

##############   Execution    ##############################################################################################################
# creates dataset to test the full operator pipeline, including the transformation matrix
combined_dataset = CombinedDataset(src_dataset, tgt_dataset, calibration_only=False) 

# fetches data from the operator. src_xs, src_ys are the source domain data, tgt_xs, tgt_ys are the target domain data.
# The xs and ys are 3-dimensional. The first dimension is a batch of functions. The next is the number of data points.
# the last is the dimensionality of the input (or output) space.
# Info is mostly useful for plotting, and is not used by the algorithm.
src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device, plot_only=False)

# compute the representation of the source data in the target domain
rep, _ = src_model.compute_representation(src_xs, src_ys, method="least_squares")

# apply the transformation matrix to the source representation
# this transforms the source representation to the target representation
# conceptually, this transforms a source function to a target function,  f -> Tf
rep = rep @transformation.T

# predict the target domain ys from the transformed source representation
y_hats = tgt_model.predict(tgt_xs, rep)

# lets see how well it did
loss = torch.nn.functional.mse_loss(y_hats, tgt_ys)
print(f"\nFull Operator Loss: {loss.item():.2e}\n")



##############   PLOT    ##################################################################################################################
print("Plotting...")
print("See logs_mwe/ for a visual representation of the results")
with torch.no_grad():
    # fetch the correct plotting functions
    plot_source = plot_source_cubic
    plot_target = plot_target_cubic_derivative
    plot_transformation = plot_transformation_derivative
    

    # plot source
    example_xs, example_ys, xs, ys, info = src_dataset.sample(device)
    info["model_type"] = f"matrix_least_squares"
    y_hats = src_model.predict_from_examples(example_xs, example_ys, xs, method="least_squares",)
    plot_source(xs, ys, y_hats, info, logdir)

    # plot target domain
    example_xs, example_ys, xs, ys, info = tgt_dataset.sample(device)
    info["model_type"] = f"matrix_least_squares" 
    y_hats = tgt_model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")
    plot_target(xs, ys, y_hats, info, logdir)


    # plot transformation for all model types
    example_xs, example_ys, xs, ys, info = combined_dataset.sample(device)
    info["model_type"] = f"matrix_least_squares" 
    example_y_hats = src_model.predict_from_examples(example_xs, example_ys, example_xs, method="least_squares")
    # full pipeline forward pass
    rep, _ = src_model.compute_representation(example_xs, example_ys, method="least_squares")
    rep = rep @transformation.T
    y_hats = tgt_model.predict(xs, rep)
    # plot
    plot_transformation(example_xs, example_ys, example_y_hats, xs, ys, y_hats, info, logdir)



