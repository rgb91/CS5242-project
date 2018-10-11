"""
Created by Sanjay at 10/5/2018

Feature: Enter feature name here
Enter feature description here
"""
from source.read_pdb import read_pdb_file
from os import listdir
from os.path import isfile, join

datapath = r'C:\Users\Sanjay Saha\CS5242-project\data100'
allfiles = [join (datapath, f) for f in listdir (datapath) if isfile (join (datapath, f))]
for i in range (0, len (allfiles), 2):
    X_list_ligand, Y_list_ligand, Z_list_ligand, atomtype_list_ligand = read_pdb_file (allfiles[i])
    X_list_protein, Y_list_protein, Z_list_protein, atomtype_list_protein = read_pdb_file (allfiles[i + 1])
