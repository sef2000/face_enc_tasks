import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

def read_mat(file_path):
    """Reads a .mat file"""
    mat_contents = sio.loadmat(file_path)
    data = mat_contents['meanFR']
    ids = mat_contents['im_code'].squeeze()
    return data.squeeze(), ids

mat_files = "/data/saskia_fohs/enc_phys/Data"
file_interest = "CelebA_Base_Cor_p20WV.mat"

data_neurons, ids = read_mat(f"{mat_files}/{file_interest}")

print(ids)

# save as npy
# has right order as neuron information; double checked by identity repetitions
np.save(f"{mat_files}/celeb_neurons.npy", data_neurons)