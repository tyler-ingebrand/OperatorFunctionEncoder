from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset

class OperatorDataset(BaseDataset):
    def __init__(self,   input_size:Tuple[int],
                         output_size:Tuple[int],
                         total_n_functions:Union[int, float] = float('inf'),
                         total_n_samples_per_function:Union[int, float] = float('inf'),
                         data_type:str = "deterministic",
                         n_functions_per_sample:int = 10,
                         n_examples_per_sample:int = 1000,
                         n_points_per_sample:int = 10000,
                         freeze_example_xs:bool = False,
                         freeze_xs:bool = False,
                         ):
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         total_n_functions=total_n_functions,
                         total_n_samples_per_function=total_n_samples_per_function,
                         data_type=data_type,
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         )
        self.freeze_example_xs = freeze_example_xs
        self.example_xs = None
        self.freeze_xs = freeze_xs
        self.frozen_xs = None

    # the info dict is used to generate data. So first we generate an info dict
    def sample_info(self) -> dict:
        pass

    # this function is used to generate the data
    def sample_inputs(self, info, n_samples) -> torch.tensor:
        pass

    def compute_outputs(self, info, inputs) -> torch.tensor:
        pass

    def sample(self, device:Union[str, torch.device]) -> Tuple[ torch.tensor,
                                                                torch.tensor,
                                                                torch.tensor,
                                                                torch.tensor,
                                                                dict]:
        # this is sampling functions
        info = self.sample_info()

        # get example input data. Whether or not its frozen depends on deeponet.
        if not self.freeze_example_xs: # sample every time
            example_xs = self.sample_inputs(info, self.n_examples_per_sample)
        elif self.example_xs is None: # sample first time
            example_xs = self.sample_inputs(info, self.n_examples_per_sample)
            # we need the examples to be the same for all functions, so grab the first function
            # then repeat it
            example_xs = example_xs[0]
            example_xs = example_xs.repeat(self.n_functions_per_sample, 1, 1)
            self.example_xs = example_xs
        else: # otherwise load it
            example_xs = self.example_xs

        # compute example output data.
        example_ys = self.compute_outputs(info, example_xs)

        # get target data
        # even for deeponet these xs are always random
        # but they may be frozen for deeponet_pod
        if not self.freeze_xs:  # sample every time
            xs = self.sample_inputs(info, self.n_points_per_sample)
        elif self.frozen_xs is None:  # sample first time
            xs = self.sample_inputs(info, self.n_points_per_sample)
            # we need the examples to be the same for all functions, so grab the first function
            # then repeat it
            xs = xs[0]
            xs = xs.repeat(self.n_functions_per_sample, 1, 1)
            self.frozen_xs = xs[0] # save just the data, we will repeat it later
        else:  # otherwise load it
            xs = self.frozen_xs.repeat(self.n_functions_per_sample, 1, 1) # repeat it

        ys = self.compute_outputs(info, xs)

        # change device
        example_xs = example_xs.to(device)
        example_ys = example_ys.to(device)
        xs = xs.to(device)
        ys = ys.to(device)

        return example_xs, example_ys, xs, ys, info

    def check_dataset(self):
        pass # we are doing exotic things, so this check does not apply.


class CombinedDataset(BaseDataset):

    def __init__(self, src_dataset:OperatorDataset, tgt_dataset:OperatorDataset, calibration_only:bool = False):
        super().__init__(input_size=(1,), # depends on if you mean src or tgt dataset. Dont use this to init a function encoder.
                         output_size=(1,),
                         total_n_functions=src_dataset.n_functions,
                         total_n_samples_per_function=src_dataset.n_samples_per_function,
                         data_type="deterministic",
                         n_functions_per_sample=src_dataset.n_functions_per_sample,
                         n_examples_per_sample=src_dataset.n_examples_per_sample,
                         n_points_per_sample=tgt_dataset.n_points_per_sample,
                         )
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.freeze_example_xs = src_dataset.freeze_example_xs
        self.freeze_xs = tgt_dataset.freeze_xs
        self.example_xs = None
        self.frozen_xs = None
        self.calibration_only = calibration_only

    def sample(self, device: Union[str, torch.device]) -> Tuple[torch.tensor,
                                                                torch.tensor,
                                                                torch.tensor,
                                                                torch.tensor,
                                                                dict]:
        # sample functions for src dataset
        info = self.src_dataset.sample_info()

        # get example input data. Whether or not its frozen depends on deeponet.
        # this comes from the src dataset
        if not self.freeze_example_xs:  # sample every time
            example_xs = self.src_dataset.sample_inputs(info, self.src_dataset.n_examples_per_sample)
        elif self.example_xs is None:  # sample first time
            example_xs = self.src_dataset.sample_inputs(info, self.src_dataset.n_examples_per_sample)
            # we need the examples to be the same for all functions, so grab the first function
            # then repeat it
            example_xs = example_xs[0]
            example_xs = example_xs.repeat(self.n_functions_per_sample, 1, 1)
            self.example_xs = example_xs[0]
        else:  # otherwise load it
            example_xs = self.example_xs.repeat(self.n_functions_per_sample, 1, 1)

        # compute example output data.
        example_ys = self.src_dataset.compute_outputs(info, example_xs)

        # get target data
        # even for deeponet these xs are always random, except for deeponet_pod
        if not self.freeze_xs:
            xs = self.tgt_dataset.sample_inputs(info, self.tgt_dataset.n_points_per_sample if not self.calibration_only else self.tgt_dataset.n_examples_per_sample)
        elif self.frozen_xs is None:
            xs = self.tgt_dataset.sample_inputs(info, self.tgt_dataset.n_points_per_sample if not self.calibration_only else self.tgt_dataset.n_examples_per_sample)
            # we need the examples to be the same for all functions, so grab the first function
            # then repeat it
            xs = xs[0]
            xs = xs.repeat(self.n_functions_per_sample, 1, 1)
            self.frozen_xs = xs[0]
        else:
            xs = self.frozen_xs.repeat(self.n_functions_per_sample, 1, 1)

        # compute target data
        ys = self.tgt_dataset.compute_outputs(info, xs)

        # change device
        example_xs = example_xs.to(device)
        example_ys = example_ys.to(device)
        xs = xs.to(device)
        ys = ys.to(device)

        return example_xs, example_ys, xs, ys, info


    def check_dataset(self):
        pass # we are doing exotic things, so this check does not apply.
