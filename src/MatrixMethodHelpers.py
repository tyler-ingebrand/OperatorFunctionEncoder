import torch
from FunctionEncoder import FunctionEncoder
from tqdm import trange

from src.Datasets.OperatorDataset import CombinedDataset


def compute_A(src_model:FunctionEncoder,
              tgt_model:FunctionEncoder,
              combined_dataset:CombinedDataset,
              device,
              train_method):
    with torch.no_grad():
        all_src_Cs = []
        all_tgt_Cs = []
        # collect a bunch of data. We have to accumulate to save memory
        for epoch in trange(100):
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
        print("Transformation error: ", torch.mean(torch.norm(tgt_Cs - src_Cs @ A.T, dim=-1)).item())
        return A