# OperatorFunctionEncoder

This is the official repo for "[Basis to Basis Operator Learning via Function Encoders](https://arxiv.org/abs/2410.00171)". See the [project page](https://tyler-ingebrand.github.io/OperatorFunctionEncoder/) for more information. 

### Getting Started
First, install torch using this website's command:
https://pytorch.org/get-started/locally/

Then, install all packages using pip:

```commandline
pip install FunctionEncoder==0.0.4 numpy matplotlib tqdm scipy tensorboard
```

Download data using the following commands:
```commandline
python src/Datasets/DarcyDataset.py
python download_data.py
```

### Working Directory
All commands are run from OperatorFunctionEncoder, the base working directory. Do not change into src/ or run_scripts/ or plotting_scripts/, this will likely break things. 

### Running the code
To run the code for one example, run the following command:
```commandline
python test.py (args)
```
Then are numerous arguments to select different algs and datasets.

Alternatively, use the following to run all experiments:
```commandline
chmod +x ./run_scripts/run_ablation_n_basis.sh # makes it executable
chmod +x ./run_scripts/run_ablation_n_sensor.sh 
chmod +x ./run_scripts/run_ablation_unfreeze.sh 
chmod +x ./run_scripts/run_all.sh 
chmod +x ./run_scripts/run_experiment.sh 
./run_scripts/run_all.sh
```
You will likely have to change the arguments at the top of the file to 1 Gpu. 

Note this will take a long time to run.

### Experimental Results

Training curves for the experiments in the paper are available [here](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi%3A10.18738%2FT8%2FE5YJOG)
