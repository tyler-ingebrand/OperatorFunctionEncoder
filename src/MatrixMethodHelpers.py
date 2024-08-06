from typing import Union

import torch
from FunctionEncoder import FunctionEncoder, BaseCallback, TensorboardCallback
from tqdm import trange

from src.Datasets.HeatDataset import HeatSrcDataset
from src.Datasets.OperatorDataset import CombinedDataset


def compute_A(src_model:FunctionEncoder,
              tgt_model:FunctionEncoder,
              combined_dataset:CombinedDataset,
              device,
              train_method,
              callback:TensorboardCallback):
    with torch.no_grad():
        all_src_Cs = []
        all_tgt_Cs = []
        # collect a bunch of data. We have to accumulate to save memory
        for epoch in range(100):
            src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device)

            # then get the representations
            src_Cs, _ = src_model.compute_representation(src_xs, src_ys, method=train_method)
            tgt_Cs, _ = tgt_model.compute_representation(tgt_xs, tgt_ys, method=train_method)

            all_src_Cs.append(src_Cs)
            all_tgt_Cs.append(tgt_Cs)

        src_Cs = torch.cat(all_src_Cs, dim=0)
        tgt_Cs = torch.cat(all_tgt_Cs, dim=0)

        # now compute the transformation via LS solution.
        A = torch.linalg.lstsq(src_Cs, tgt_Cs).solution.T
        loss = torch.mean(torch.norm(tgt_Cs - src_Cs @ A.T, dim=-1)).item()
        callback.tensorboard.add_scalar("transformation/loss", loss, callback.total_epochs)
        return A

def train_nonlinear_transformation(A:torch.nn.Sequential,
                                   opt:torch.optim.Optimizer,
                                   src_basis:FunctionEncoder,
                                   tgt_basis:FunctionEncoder,
                                   train_method:str,
                                   combined_dataset:CombinedDataset,
                                   epochs:int,
                                   callback:TensorboardCallback):
    # increase the number of functions per sample, as these are data points for training the nn
    old_n_functions_src = combined_dataset.src_dataset.n_functions_per_sample
    old_n_functions_tgt = combined_dataset.tgt_dataset.n_functions_per_sample
    combined_dataset.src_dataset.n_functions_per_sample = 128
    combined_dataset.tgt_dataset.n_functions_per_sample = 128


    device = next(A.parameters()).device
    for epoch in range(epochs):
        with torch.no_grad():
            src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device)

            # then get the representations
            if type(combined_dataset.src_dataset) != HeatSrcDataset:
                src_Cs, _ = src_basis.compute_representation(src_xs, src_ys, method=train_method)
            else:
                src_Cs = src_ys[:, 0, :] # fetches temperature and alpha
            tgt_Cs, _ = tgt_basis.compute_representation(tgt_xs, tgt_ys, method=train_method)

        # now compute the transformation via LS solution.
        tgt_Cs_hat = A(src_Cs)

        # now train the transformation
        opt.zero_grad()
        loss = torch.mean(torch.norm(tgt_Cs - tgt_Cs_hat, dim=-1))
        loss.backward()
        opt.step()

        # log the loss
        callback.tensorboard.add_scalar("transformation/loss", loss.item(), callback.total_epochs - epochs + epoch)

    # reset the number of functions per sample
    combined_dataset.src_dataset.n_functions_per_sample = old_n_functions_src
    combined_dataset.tgt_dataset.n_functions_per_sample = old_n_functions_tgt

    return A, opt


def get_num_parameters(model:Union[torch.nn.Module, dict, torch.tensor]):
    if isinstance(model, dict):
        num_params = sum([get_num_parameters(v) for v in model.values()])
    elif isinstance(model, torch.nn.Module):
        num_params = sum([torch.numel(p) for p in model.parameters()])
    elif isinstance(model, torch.Tensor):
        num_params = torch.numel(model)
    elif isinstance(model, type(None)):
        num_params = 0
    else:
        raise ValueError(f"Model type {type(model)} not supported")
    return num_params

def get_num_layers(model:torch.nn.Module):
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            count += 1
    return count

def predict_number_params(model_type:str,
                          n_sensors,
                          n_basis,
                          hidden_size,
                          n_layers,
                          src_input_space,
                          src_output_space,
                          tgt_input_space,
                          tgt_output_space,
                          transformation_type,
                          dataset_type):
    num_params = 0
    if model_type == "SVD":
        # src model
        input_size = src_input_space[0]  # only 1D input supported for now
        output_size = src_output_space[0] * n_basis
        num_params += input_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * output_size + output_size

        # tgt model
        input_size = tgt_input_space[0]
        output_size = tgt_output_space[0] * n_basis
        num_params += input_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * output_size + output_size

        # Sigma values
        num_params += n_basis
    elif model_type == "Eigen":
        # src and tgt model (they are shared)
        input_size = src_input_space[0]  # only 1D input supported for now
        output_size = src_output_space[0] * n_basis
        num_params += input_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * output_size + output_size

        # Sigma values
        num_params += n_basis

    elif model_type == "deeponet":
        # branch
        num_params += src_output_space[0] * n_sensors * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * tgt_output_space[0] * n_basis + tgt_output_space[0] * n_basis

        # trunk
        num_params += tgt_input_space[0] * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * tgt_output_space[0] * n_basis + tgt_output_space[0] * n_basis

        # bias
        num_params += tgt_output_space[0]
    elif model_type == "matrix":

        # src function encoder
        if dataset_type != "Heat": # heat has no src function encoder
            input_size = src_input_space[0]
            output_size = src_output_space[0] * n_basis
            num_params += input_size * hidden_size + hidden_size
            num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
            num_params += hidden_size * output_size + output_size

        # tgt function encoder
        input_size = tgt_input_space[0]
        output_size = tgt_output_space[0] * (n_basis + 1) # note the plus 1 is for debugging purposes
        num_params += input_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * output_size + output_size

        # operator A
        if transformation_type == "linear":
            num_params += n_basis * (n_basis+1)
        else:
            input_size = n_basis if dataset_type != "Heat" else 2 # alpha and temperature
            num_params += input_size * hidden_size + hidden_size
            num_params += (hidden_size * hidden_size + hidden_size) * (n_layers - 2)
            num_params += hidden_size * (1+n_basis) + (1+n_basis)
    elif model_type == "deeponet_cnn":
        # branch
        num_params += 16 * 2 * 3 * 3 + 16
        num_params += 32 * 16 * 3 * 3 + 32
        num_params += 64 * 32 * 3 * 3 + 64
        num_params += 3136 * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * tgt_output_space[0] * n_basis + tgt_output_space[0] * n_basis

        # trunk
        num_params += tgt_input_space[0] * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * tgt_output_space[0] * n_basis + tgt_output_space[0] * n_basis

        # bias
        num_params += tgt_output_space[0]
    else:
        raise ValueError(f"Model type {model_type} not supported")

    return num_params


# for a given set of hyperparameters, gets the hidden layer size that comes most closely to the target number of parameters
def get_hidden_layer_size(target_n_parameters:int,
                          model_type:str,
                          n_sensors,
                          n_basis,
                          n_layers,
                          src_input_space,
                          src_output_space,
                          tgt_input_space,
                          tgt_output_space,
                          transformation_type,
                          dataset_type):
    def loss_function(hidden_size):
        return abs(predict_number_params(model_type,
                                            n_sensors,
                                            n_basis,
                                            hidden_size,
                                            n_layers,
                                            src_input_space,
                                            src_output_space,
                                            tgt_input_space,
                                            tgt_output_space,
                                            transformation_type,
                                            dataset_type) - target_n_parameters)

    def ternary_search(start, end):
        while end - start > 2:
            mid1 = start + (end - start) // 3
            mid2 = end - (end - start) // 3

            loss1 = loss_function(mid1)
            loss2 = loss_function(mid2)

            if loss1 < loss2:
                end = mid2
            else:
                start = mid1

        best_input = start
        min_loss = loss_function(start)
        for x in range(start, end + 1):
            current_loss = loss_function(x)
            if current_loss < min_loss:
                min_loss = current_loss
                best_input = x

        return best_input, min_loss

    start = 10
    end = int(1e5)

    best_input, min_loss = ternary_search(start, end)
    # print(f'The best input is {best_input} with a loss of {min_loss}')
    # print(f"Target number of parameters: {target_n_parameters}")
    # print(f"Predicted number of parameters: {predict_number_params(model_type, n_sensors, n_basis, best_input, n_layers, src_input_space, src_output_space, tgt_input_space, tgt_output_space, transformation_type)}")
    return best_input