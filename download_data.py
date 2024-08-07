from onedrivedownloader import download
import urllib.request
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
        count = 0



# download the files
print("Downloading datasets to src/Datasets/...")

# #### Heat ##############################################################################################################################################
ln = "https://drive.usercontent.google.com/download?id=1mSMplhocRU_0MJSwAnlXYpo4SCYoSkPk&export=download&authuser=0&confirm=t&uuid=7ef4e753-bfc3-4855-92a2-1ea57e39bf3f&at=APZUnTX9xjPEoRhinxBuhPCVk4iG:1723049906103"
urllib.request.urlretrieve(ln, "src/Datasets/Heat/heatequation_train.mat", reporthook=show_progress)

ln = "https://drive.usercontent.google.com/download?id=1JUFD9VDa2Drgd9OZ7yDAGapQ8AJKeaC1&export=download&authuser=0&confirm=t&uuid=9699c394-dae0-45e3-97ba-d5c0462695b8&at=APZUnTXCAZ_BaRILl8N2-q5n2tS_:1723050016071"
urllib.request.urlretrieve(ln, "src/Datasets/Heat/heatequation_test.mat", reporthook=show_progress)


# #### L-shaped, linear Darcy ##############################################################################################################################
# ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=14146ad4%2D794c%2D4f40%2Db91b%2D4f2c1f6f7c33"
# download(ln, filename="src/Datasets/L-Shaped/linearDarcy_train.mat")
#
# ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=45a249fe%2D0da1%2D4a79%2Db95e%2De0e318b35b01"
# download(ln, filename="src/Datasets/L-Shaped/linearDarcy_test.mat")
#
# #### 1D, nonlinear Darcy ################################################################################################################################
# ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=30807c30%2Dab4b%2D46a9%2Dbfb6%2D0c3cd6535f2b"
# download(ln, filename="src/Datasets/Darcy/nonlinearDarcy_train.mat")
#
# ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=de04f1a6%2D8144%2D4585%2D906a%2D7a6614776dcd"
# download(ln, filename="src/Datasets/Darcy/nonlinearDarcy_test.mat")
#
# #### Elastic ############################################################################################################################################
# ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fsgoswam4%5Fjh%5Fedu%2FDocuments%2FShared%20with%20others%2Fcomparing%5Fwith%5Ffunction%5Fencoders%2Flinear%5Felasticity%2FlinearElasticity%5Ftrain%2Emat"
# download(ln, filename="src/Datasets/Elastic/linearElasticity_train.mat")
#
# ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=9ff2049f%2D16c4%2D4727%2Dbe48%2D058bef59384c"
# download(ln, filename="src/Datasets/Elastic/linearElasticity_test.mat")
#
# print("Done!\n\n")