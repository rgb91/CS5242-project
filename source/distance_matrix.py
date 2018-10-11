"""
Created by Sanjay at 10/5/2018

Feature: Enter feature name here
Enter feature description here
"""
from source.read_pdb import read_pdb_file
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.spatial import distance_matrix

datapath = r'C:\Users\Sanjay Saha\CS5242-project\data100'
allfiles = [join(datapath, f) for f in listdir(datapath) if isfile(join(datapath, f))]

X_list, Y_list, Z_list, atomtype_list = read_pdb_file (allfiles[0])
X_list = np.reshape (np.array (X_list), (np.array (X_list).shape[0], 1))
Y_list = np.reshape (np.array (Y_list), (np.array (Y_list).shape[0], 1))
Z_list = np.reshape (np.array (Z_list), (np.array (Z_list).shape[0], 1))
coordinates = np.asarray(np.concatenate ((X_list, Y_list, Z_list), axis=1), dtype='float')
distance_matrix = distance_matrix(coordinates, coordinates)
print(distance_matrix)

# for f in allfiles:
#     X_list, Y_list, Z_list, atomtype_list = read_pdb_file (f)
#     X_list = np.reshape (np.array (X_list), (np.array (X_list).shape[0], 1))
#     Y_list = np.reshape (np.array (Y_list), (np.array (Y_list).shape[0], 1))
#     Z_list = np.reshape (np.array (Z_list), (np.array (Z_list).shape[0], 1))
#     coordinates = np.asarray(np.concatenate ((X_list, Y_list, Z_list), axis=1), dtype='float')
#     distance_matrix = distance_matrix(coordinates, coordinates)