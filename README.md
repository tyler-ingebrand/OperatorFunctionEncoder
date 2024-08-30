# OperatorFunctionEncoder

This is the official repo for ...

### Getting Started
First, install torch using this website's command:
https://pytorch.org/get-started/locally/

Then, install all packages using pip:

```commandline
pip install FunctionEncoder==0.0.4 numpy matplotlib tqdm scipy tensorboard
```

Download data using the following command:
```commandline
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
chmod +x ./run_scripts/run_experiment.sh # makes it executable
./run_scripts/run_experiment.sh
```
You will likely have to change the arguments at the top of the file to 1 Gpu, 1 process per GPU. 
