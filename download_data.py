from onedrivedownloader import download

# download the files
print("Downloading datasets")

#### Heat ##############################################################################################################################################
ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=447af3bb%2D836c%2D4eb6%2Db5d4%2D977d982f4860"
download(ln, filename="src/Datasets/Heat/heatequation_train.mat")

ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=c7e463e4%2Dd397%2D457b%2D89f6%2D0bd760c63697"
download(ln, filename="src/Datasets/Heat/heatequation_test.mat")


#### L-shaped, linear Darcy ##############################################################################################################################
ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=14146ad4%2D794c%2D4f40%2Db91b%2D4f2c1f6f7c33"
download(ln, filename="src/Datasets/L-Shaped/linearDarcy_train.mat")

ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=45a249fe%2D0da1%2D4a79%2Db95e%2De0e318b35b01"
download(ln, filename="src/Datasets/L-Shaped/linearDarcy_test.mat")

#### 1D, nonlinear Darcy ################################################################################################################################
ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=30807c30%2Dab4b%2D46a9%2Dbfb6%2D0c3cd6535f2b"
download(ln, filename="src/Datasets/Darcy/nonlinearDarcy_train.mat")

ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=de04f1a6%2D8144%2D4585%2D906a%2D7a6614776dcd"
download(ln, filename="src/Datasets/Darcy/nonlinearDarcy_test.mat")

#### Elastic ############################################################################################################################################
ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=9e212282%2D54e4%2D4ae4%2D9032%2Dbdfd11313810"
download(ln, filename="src/Datasets/Elastic/linearElasticity_train.mat")

ln = "https://livejohnshopkins-my.sharepoint.com/personal/sgoswam4_jh_edu/_layouts/15/download.aspx?UniqueId=9ff2049f%2D16c4%2D4727%2Dbe48%2D058bef59384c"
download(ln, filename="src/Datasets/Elastic/linearElasticity_test.mat")

print("Done!\n\n\n")