import urllib.request
import zipfile
import os

import numpy as np
import tqdm

pbar, count = None, 0
def show_progress(block_num, block_size, total_size):
    global pbar
    global count
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True)
    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded - count)
        count = downloaded
    else:
        pbar.close()
        pbar = None

# download the zip at this link
url = "https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1"
print("Downloading...")
urllib.request.urlretrieve(url, "src/Datasets/FluidFlow/DeepCFD.zip", reporthook=show_progress)

# extract the zip file
print("Extracting...")
with zipfile.ZipFile("src/Datasets/FluidFlow/DeepCFD.zip", 'r') as zip_ref:
    zip_ref.extractall("src/Datasets/FluidFlow")

# remove the zip file
print("Cleaning up...")
os.remove("src/Datasets/FluidFlow/DeepCFD.zip")
os.system("rm -r src/Datasets/FluidFlow/Models")
os.remove("src/Datasets/FluidFlow/DeepCFD.py")
os.remove("src/Datasets/FluidFlow/functions.py")
os.remove("src/Datasets/FluidFlow/pytorchtools.py")
os.remove("src/Datasets/FluidFlow/train_functions.py")

# format the data into train and test sets.
# load data
print("Formatting data...")
dataX = np.load('./src/Datasets/FluidFlow/dataX.pkl', allow_pickle=True)
dataY = np.load('./src/Datasets/FluidFlow/dataY.pkl', allow_pickle=True)

# compute size of train test split
num_functions = dataX.shape[0]
train_size = int(num_functions * 0.8)

# split data
train_dataX = dataX[:train_size]
train_dataY = dataY[:train_size]
test_dataX = dataX[train_size:]
test_dataY = dataY[train_size:]

# save data
np.save('./src/Datasets/FluidFlow/train_dataX.npy', train_dataX)
np.save('./src/Datasets/FluidFlow/train_dataY.npy', train_dataY)
np.save('./src/Datasets/FluidFlow/test_dataX.npy', test_dataX)
np.save('./src/Datasets/FluidFlow/test_dataY.npy', test_dataY)

# remove the old data
os.remove('./src/Datasets/FluidFlow/dataX.pkl')
os.remove('./src/Datasets/FluidFlow/dataY.pkl')

print("Done!\n\n")