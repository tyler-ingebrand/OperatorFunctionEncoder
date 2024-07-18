from typing import Union, Tuple

import torch
from FunctionEncoder import FunctionEncoder, BaseDataset, BaseCallback
from FunctionEncoder.Model.Architecture.Euclidean import Euclidean
from FunctionEncoder.Model.Architecture.MLP import MLP
from tqdm import trange


class SVDEncoder(FunctionEncoder):
    def __init__(self,
                 input_size_src: tuple[int],
                 output_size_src: tuple[int],
                 input_size_tgt: tuple[int],
                 output_size_tgt: tuple[int],
                 data_type: str,
                 n_basis: int = 100,
                 model_type: str = "MLP",
                 model_kwargs: dict = dict(),
                 method: str = "least_squares",
                 use_eigen_decomp=False,
                 ):
        assert len(input_size_src) == 1, "Only 1D input supported for now"
        assert input_size_src[0] >= 1, "Input size must be at least 1"
        assert len(input_size_tgt) == 1, "Only 1D input supported for now"
        assert input_size_tgt[0] >= 1, "Input size must be at least 1"
        assert len(output_size_src) == 1, "Only 1D output supported for now"
        assert output_size_src[0] >= 1, "Output size must be at least 1"
        assert len(output_size_tgt) == 1, "Only 1D output supported for now"
        assert output_size_tgt[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic"], f"Unknown data type: {data_type}"
        super(FunctionEncoder, self).__init__()

        # hyperparameters
        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt
        self.n_basis = n_basis
        self.method = method
        self.data_type = data_type
        self.use_eigen_decomp = use_eigen_decomp

        # models and optimizers
        # note src_model is V in SVD, tgt_model is U in SVD, sigma_values is the singular values
        self.src_model = self._build(model_type, model_kwargs, input_size_src, output_size_src)
        self.sigma_values = torch.nn.parameter.Parameter(torch.randn(self.n_basis) * 0.1)
        if not self.use_eigen_decomp:
            self.tgt_model = self._build(model_type, model_kwargs, input_size_tgt, output_size_tgt)
            params = [*self.src_model.parameters()] + [self.sigma_values] + [*self.tgt_model.parameters()]
        else:
            self.tgt_model = self.src_model
            params = [*self.src_model.parameters()] + [self.sigma_values]
        self.opt = torch.optim.Adam(params, lr=1e-3)

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.average_function = None # this is legacy from function encoder, we dont use it.


    def _build(self,
               model_type: str,
               model_kwargs: dict,
               input_size,
               output_size,
               average_function: bool = False) -> torch.nn.Module:
        # the average function is basically a single function, so n_basis is 1
        if average_function:
            n_basis = 1
        else:
            n_basis = self.n_basis

        if model_type == "MLP":
            return MLP(input_size=input_size,
                       output_size=output_size,
                       n_basis=n_basis,
                       **model_kwargs)
        elif model_type == "Euclidean":
            return Euclidean(input_size=input_size,
                             output_size=output_size,
                             n_basis=n_basis,
                             **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def compute_representation(self,
                               example_xs: torch.tensor,
                               example_ys: torch.tensor,
                               method: str = "inner_product",
                               representation_dataset="source",
                               prediction_dataset="target",
                               **kwargs) -> Tuple[torch.tensor, Union[torch.tensor, None]]:

        if representation_dataset == "source":
            assert example_xs.shape[-len( self.input_size_src):] == self.input_size_src, f"example_xs must have shape (..., {self.input_size_src}). Expected {self.input_size_src}, got {example_xs.shape[-len(self.input_size_src):]}"
            assert example_ys.shape[-len(self.output_size_src):] == self.output_size_src, f"example_ys must have shape (..., {self.output_size_src}). Expected {self.output_size_src}, got {example_ys.shape[-len(self.output_size_src):]}"
        else:
            assert example_xs.shape[-len(self.input_size_tgt):] == self.input_size_tgt, f"example_xs must have shape (..., {self.input_size_tgt}). Expected {self.input_size_tgt}, got {example_xs.shape[-len(self.input_size_tgt):]}"
            assert example_ys.shape[-len(self.output_size_tgt):] == self.output_size_tgt, f"example_ys must have shape (..., {self.output_size_tgt}). Expected {self.output_size_tgt}, got {example_ys.shape[-len(self.output_size_tgt):]}"
        assert example_xs.shape[:-len(self.input_size_src)] == example_ys.shape[:-len(self.output_size_src)], f"example_xs and example_ys must have the same shape except for the last {len(self.input_size)} dimensions. Expected {example_xs.shape[:-len(self.input_size)]}, got {example_ys.shape[:-len(self.output_size)]}"

        # if not in terms of functions, add a function batch dimension
        reshaped = False
        if len(example_xs.shape) - len(self.input_size_src) == 1:
            reshaped = True
            example_xs = example_xs.unsqueeze(0)
            example_ys = example_ys.unsqueeze(0)

        # compute representation
        if representation_dataset == "source":
            Gs = self.src_model.forward(example_xs)  # forward pass of the basis functions. Uses src basis functions.
        else:
            Gs = self.tgt_model.forward(example_xs)

        if method == "inner_product":
            representation = self._compute_inner_product_representation(Gs, example_ys)
            gram = None
        elif method == "least_squares":
            representation, gram = self._compute_least_squares_representation(Gs, example_ys, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # reshape if necessary
        if reshaped:
            assert representation.shape[0] == 1, "Expected a single function batch dimension"
            representation = representation.squeeze(0)
        return representation, gram

    def predict(self,
                xs: torch.tensor,
                representations: torch.tensor,
                precomputed_average_ys: Union[torch.tensor, None] = None,
                representation_dataset="source",
                prediction_dataset="target") -> torch.tensor:

        assert len(xs.shape) == 3, f"Expected xs to have shape (f,d,n), got {xs.shape}"
        assert len(representations.shape) == 2, f"Expected representations to have shape (f,k), got {representations.shape}"
        assert xs.shape[0] == representations.shape[0], f"Expected xs and representations to have the same number of functions, got {xs.shape[0]} and {representations.shape[0]}"

        # this is weighted combination of basis functions
        if representation_dataset == "source" and prediction_dataset == "target": # this is the usual method, where the target model is used
            Gs = self.tgt_model.forward(xs)
            y_hats = torch.einsum("fdmk,fk,k->fdm", Gs, representations, self.sigma_values)
        elif representation_dataset == "source" and prediction_dataset == "source": # this is used only for plots, to see if the src basis spans the space
            Gs = self.src_model.forward(xs)
            y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)
        elif representation_dataset == "target" and prediction_dataset == "target": # this is used only for plots, to see if the src basis spans the space
            Gs = self.tgt_model.forward(xs)
            y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)
        else:
            raise ValueError(f"Unknown dataset combintaion: {representation_dataset}, {prediction_dataset}")

        return y_hats

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor,
                              method: str = "inner_product",
                              representation_dataset="source",
                              prediction_dataset="target",
                              **kwargs):

        assert len(example_xs.shape) == 3, f"Expected example_xs to have shape (f,d,n), got {example_xs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert len(xs.shape) == 3, f"Expected xs to have shape (f,d,n), got {xs.shape}"
        if representation_dataset == "source":
            assert example_xs.shape[-len(self.input_size_src):] == self.input_size_src, f"Expected example_xs to have shape (..., {self.input_size_src}), got {example_xs.shape[-1]}"
            assert example_ys.shape[-len(self.output_size_src):] == self.output_size_src, f"Expected example_ys to have shape (..., {self.output_size_src}), got {example_ys.shape[-1]}"
        else:
            assert example_xs.shape[-len(self.input_size_tgt):] == self.input_size_tgt, f"Expected example_xs to have shape (..., {self.input_size_tgt}), got {example_xs.shape[-1]}"
            assert example_ys.shape[-len(self.output_size_tgt):] == self.output_size_tgt, f"Expected example_ys to have shape (..., {self.output_size_tgt}), got {example_ys.shape[-1]}"
        if prediction_dataset == "source":
            assert xs.shape[-len(self.input_size_src):] == self.input_size_src, f"Expected xs to have shape (..., {self.input_size_src}), got {xs.shape[-1]}"
        else:
            assert xs.shape[-len(self.input_size_tgt):] == self.input_size_tgt, f"Expected xs to have shape (..., {self.input_size_tgt}), got {xs.shape[-1]}"
        assert example_xs.shape[0] == example_ys.shape[ 0], f"Expected example_xs and example_ys to have the same number of functions, got {example_xs.shape[0]} and {example_ys.shape[0]}"
        assert example_xs.shape[1] == example_xs.shape[ 1], f"Expected example_xs and example_ys to have the same number of datapoints, got {example_xs.shape[1]} and {example_ys.shape[1]}"
        assert example_xs.shape[0] == xs.shape[0], f"Expected example_xs and xs to have the same number of functions, got {example_xs.shape[0]} and {xs.shape[0]}"

        representations, _ = self.compute_representation(example_xs, example_ys, method=method, representation_dataset=representation_dataset, prediction_dataset =prediction_dataset, **kwargs)
        y_hats = self.predict(xs, representations, representation_dataset=representation_dataset, prediction_dataset =prediction_dataset,)
        return y_hats

    def _param_string(self):
        params = {}
        params["input_size_src"] = self.input_size_src
        params["output_size_src"] = self.output_size_src
        params["input_size_tgt"] = self.input_size_tgt
        params["output_size_tgt"] = self.output_size_tgt
        params["n_basis"] = self.n_basis
        params["method"] = self.method
        params["model_type"] = self.model_type
        for k, v in self.model_kwargs.items():
            params[k] = v
        params = {k: str(v) for k, v in params.items()}
        return params

    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback: BaseCallback = None):

        # verify dataset is correct
        # check_dataset(dataset) # TODO this is the only lined that changed, since our dataset has a different defintion

        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        # method to use for representation during training
        assert self.method in ["inner_product", "least_squares"], f"Unknown method: {self.method}"

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, xs, ys, _ = dataset.sample(device=device)

            # train average function, if it exists
            if self.average_function is not None:
                # predict averages
                expected_yhats = self.average_function.forward(xs)

                # compute average function loss
                average_function_loss = self._distance(expected_yhats, ys).square().mean()

                # we only train average function to fit data in general, so block backprop from the basis function loss
                expected_yhats = expected_yhats.detach()
            else:
                expected_yhats = None

            # approximate functions, compute error
            representation, gram = self.compute_representation(example_xs, example_ys, method=self.method)
            y_hats = self.predict(xs, representation, precomputed_average_ys=expected_yhats)
            prediction_loss = self._distance(y_hats, ys).square().mean()

            # LS requires regularization since it does not care about the scale of basis
            # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
            # from going to infinity.
            if self.method == "least_squares":
                norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1) ** 2).mean()

            # add loss components
            loss = prediction_loss
            if self.method == "least_squares":
                loss = loss + norm_loss
            if self.average_function is not None:
                loss = loss + average_function_loss

            # backprop with gradient clipping
            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())