import pandas
import glob
import matplotlib.pyplot as plt

# read txt
path = "/raid/shared/datasets/visoin/CelebA/Anno/identity_CelebA.txt"
df = pandas.read_csv(path, sep=" ", header=None)

# max val
max_val = df[1].max()
print(f"Max value in the second column: {max_val}")

path = "/data/saskia_fohs/enc_phys/CelebA_face_new"
all_ims = glob.glob(f"{path}/*.jpg")
# get all image names without path and extension
all_ims = [im.split("/")[-1] for im in all_ims]

# only keep rows in df, where first row is in all_ims
df = df[df[0].isin(all_ims)]

# print last ten rows
print(df.tail(10))