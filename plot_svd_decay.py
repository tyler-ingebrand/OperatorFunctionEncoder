




import os
import torch

from src.SVDEncoder import SVDEncoder
import matplotlib.pyplot as plt


experiment_dir = "old"
dataset = "Derivative"
alg = "SVD_least_squares"
load_dir = f"{experiment_dir}/{dataset}/{alg}"

# we are going to load all the eigen exampes, and print the calculated eigen values. 
for subdir in os.listdir(load_dir):
    if not os.path.isdir(f"{load_dir}/{subdir}"):
        continue
    # load the hyper params to generate the model
    params = torch.load(f"{load_dir}/{subdir}/params.pth", weights_only=True)
    n_layers, hidden_size, n_basis = params["n_layers"], params["hidden_size"], params["n_basis"]
    
    
    # the model we will load everything into 
    model = SVDEncoder(input_size_src=(1,),
                    output_size_src=(1,),
                    input_size_tgt=(1,),
                    output_size_tgt=(1,),
                    data_type="deterministic", # we dont support stochastic for now, though its possible.
                    n_basis=n_basis,
                    method="least_squares",
                    use_eigen_decomp=False,
                    model_kwargs={"n_layers":n_layers, "hidden_size":hidden_size})

    # load model
    model.load_state_dict(torch.load(f"{load_dir}/{subdir}/model.pth", weights_only=True))

    # fetch eigen values
    sigma_values = model.sigma_values.detach()

    # take abs since sign is arbitrary
    sigma_values = torch.abs(sigma_values)

    # sort them in descending order
    sigma_values = sigma_values.sort(descending=True)[0]

    # plot the decay of sigma values
    plt.plot(sigma_values)
    
plt.xlabel("Singular Value Index")
plt.ylabel("Absolute Singular Value")
plt.title("Decay of Singular Values")
print(f"Saving plot to {load_dir}/sigma_decay.png")
plt.savefig(f"{load_dir}/sigma_decay.png")