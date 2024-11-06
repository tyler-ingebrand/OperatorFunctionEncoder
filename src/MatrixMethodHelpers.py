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
                                   callback:TensorboardCallback,
                                   model_type:str):
    new_n_functions = 128 if src_basis is not None else 15 # deeponet_2staged cannot use so many

    # increase the number of functions per sample, as these are data points for training the nn
    old_n_functions_src = combined_dataset.src_dataset.n_functions_per_sample
    old_n_functions_tgt = combined_dataset.tgt_dataset.n_functions_per_sample
    old_n_functions_combined = combined_dataset.n_functions_per_sample
    combined_dataset.src_dataset.n_functions_per_sample = new_n_functions
    combined_dataset.tgt_dataset.n_functions_per_sample = new_n_functions
    combined_dataset.n_functions_per_sample = new_n_functions

    # note if src_basis is None, then its deeponet_2stage
    # then the A nn takes as input all sensors.

    device = next(A.parameters()).device
    for epoch in range(epochs):
        with torch.no_grad():
            src_xs, src_ys, tgt_xs, tgt_ys, info = combined_dataset.sample(device)

            # then get the representations
            if model_type in ["deeponet_2stage", "deeponet_2stage_cnn"]:
                if model_type == "deeponet_2stage":
                    src_Cs = src_ys.reshape(src_ys.shape[0], -1)
                else: # deeponet_2stage_cnn
                    src_Cs = src_ys

                # 2 stage deeponet, from shin et al, has a specific target representation
                # its computed via a QR factorization of the target basis
                # Note we grab [0] because the first dimension corresponds to a particular funtion
                # but all functions are required to have the same input sensors for this alg
                target_basis_matrix = tgt_basis.model(tgt_xs[0])
                Q, R_star = torch.linalg.qr(target_basis_matrix[:, 0, :]) # only works for 1d outputs
                T = torch.linalg.inv(R_star)

                # Need to compute the target representation
                A_star, _ = tgt_basis.compute_representation(tgt_xs, tgt_ys, method=train_method)

                # in 2 stage deeponet, target values are R * target representation
                tgt_Cs = torch.einsum("kl,fl->fk", R_star, A_star)

                # without QR factorization
                # T = torch.eye(tgt_basis.n_basis, device=device)
                # tgt_Cs = A_star

            else:
                if type(combined_dataset.src_dataset) != HeatSrcDataset:
                    src_Cs, _ = src_basis.compute_representation(src_xs, src_ys, method=train_method)
                else:
                    src_Cs = src_ys[:, 0, :] # fetches temperature and alpha
                tgt_Cs, _ = tgt_basis.compute_representation(tgt_xs, tgt_ys, method=train_method)
                T = None

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
    combined_dataset.n_functions_per_sample = old_n_functions_combined

    return A, opt, T


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
    elif model_type == "deeponet_pod":
        # branch
        num_params += src_output_space[0] * n_sensors * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * tgt_output_space[0] * n_basis + tgt_output_space[0] * n_basis


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
        output_size = tgt_output_space[0] * (n_basis+1) # note the plus 1 is for debugging purposes
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
            num_params += hidden_size * (n_basis+1) + (n_basis+1)
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
    elif model_type == "deeponet_2stage":
        # tgt function encoder
        input_size = tgt_input_space[0]
        output_size = tgt_output_space[0] * n_basis
        num_params += input_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * output_size + output_size

        # Branch network, also equivalent to A
        transformation_input_size = n_sensors * src_output_space[0]
        num_params += transformation_input_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * n_basis + n_basis

        # T matrix, which is not technically learned, but it is computed and updated repeatedly
        num_params += n_basis * n_basis

    elif model_type == "deeponet_2stage_cnn":
        # tgt function encoder
        input_size = tgt_input_space[0]
        output_size = tgt_output_space[0] * n_basis
        num_params += input_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * output_size + output_size

        # Branch network, also equivalent to A
        # is cnn
        num_params += 16 * 2 * 3 * 3 + 16
        num_params += 32 * 16 * 3 * 3 + 32
        num_params += 64 * 32 * 3 * 3 + 64
        num_params += 3136 * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * tgt_output_space[0] * n_basis + tgt_output_space[0] * n_basis

        # T matrix, which is not technically learned, but it is computed and updated repeatedly
        num_params += n_basis * n_basis

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


def check_parameters(args):
    # cancel eigen on non-self-adjoint operators
    if args.model_type == "Eigen" and args.dataset_type not in ["QuadraticSin", "Derivative", "Integral"]:
        print(f"Eigen can only handle self-adjoint operators, and {args.dataset_type} is not self-adjoint. Terminating.")
        exit(0)
    if args.model_type == "deeponet_2stage" and args.dataset_type == "Heat":
        print(f"DeepOnet 2 stage, applied to Heat, is the same as our method. Terminating.")
        exit(0)

    if args.model_type == "deeponet_2stage" and args.dataset_type in ["MountainCar", "Elastic", "Heat"]:
        print(f"DeepOnet 2 stage is only applicable to 1D problems. {args.dataset_type} is not 1D. Terminating.")
        exit(0)

    if args.model_type == "deeponet_cnn" and args.dataset_type in ["QuadraticSin", "Derivative", "Integral", "MountainCar", "Elastic", "Darcy", "Heat", "Burger"]:
        print(f"DeepOnet CNN is only applicable to problems where the input sensors are an image. {args.dataset_type} is not an image. The only dataset this works on is LShaped. Terminating.")
        exit(0)

    if args.model_type == "deeponet_2stage_cnn" and args.dataset_type != "LShaped":
        print(f"DeepOnet 2 stage CNN is only applicable to LShaped Dataset, not {args.dataset_type}. Terminating.")
        exit(0)