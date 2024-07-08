from typing import Union

import torch
from FunctionEncoder import FunctionEncoder


class OperatorEncoder(FunctionEncoder):
    def __init__(self,
                 input_size: tuple[int],
                 output_size: tuple[int],
                 data_type: str,
                 n_basis: int = 100,
                 model_type: str = "MLP",
                 model_kwargs: dict = dict(),
                 method: str = "least_squares",
                 use_residuals_method: bool = False,
                 ):
        assert len(input_size) == 1, "Only 1D input supported for now"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic"], f"Unknown data type: {data_type}"
        super(FunctionEncoder, self).__init__()

        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.method = method
        self.data_type = data_type

        # models and optimizers
        self.model = self._build(model_type, model_kwargs)
        self.average_function = self._build(model_type, model_kwargs,
                                            average_function=True) if use_residuals_method else None
        self.eigen_values = torch.nn.parameter.Parameter(torch.randn(n_basis) * 0.1)
        params = [*self.model.parameters()] + [self.eigen_values] # NOTE: This line is different
        if self.average_function is not None:
            params += [*self.average_function.parameters()]
        self.opt = torch.optim.Adam(params, lr=1e-3)

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs

    def predict(self,
                xs: torch.tensor,
                representations: torch.tensor,
                precomputed_average_ys: Union[torch.tensor, None] = None) -> torch.tensor:

        assert len(xs.shape) == 3, f"Expected xs to have shape (f,d,n), got {xs.shape}"
        assert len(representations.shape) == 2, f"Expected representations to have shape (f,k), got {representations.shape}"
        assert xs.shape[0] == representations.shape[0], f"Expected xs and representations to have the same number of functions, got {xs.shape[0]} and {representations.shape[0]}"

        # this is weighted combination of basis functions
        Gs = self.model.forward(xs)
        y_hats = torch.einsum("fdmk,fk,k->fdm", Gs, representations, self.eigen_values) # NOTE: This line is different

        # optionally add the average function
        # it is allowed to be precomputed, which is helpful for training
        # otherwise, compute it
        if self.average_function:
            if precomputed_average_ys is not None:
                average_ys = precomputed_average_ys
            else:
                average_ys = self.average_function.forward(xs)
            y_hats = y_hats + average_ys
        return y_hats