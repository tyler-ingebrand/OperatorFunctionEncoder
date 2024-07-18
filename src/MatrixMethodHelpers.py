import torch
from FunctionEncoder import FunctionEncoder, BaseCallback, TensorboardCallback
from tqdm import trange

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
            src_Cs, _ = src_basis.compute_representation(src_xs, src_ys, method=train_method)
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
