"""
Created by Sanjay at 10/12/2018

Feature: Utility functions are
kept here
"""
import numpy as np


def dataset_reader():
    return [],[],[],[],[]


def read_pdb_file(filename):
    with open (filename, 'r') as file:
        strline_L = file.readlines ()
    # print(strline_L)

    X_list = list ()
    Y_list = list ()
    Z_list = list ()
    atomtype_list = list ()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip ()

        line_length = len (stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length != 78:
            print ("ERROR: line length is different. Expected=78, current={}".format (line_length))

        X_list.append (float (stripped_line[30:38].strip ()))
        Y_list.append (float (stripped_line[38:46].strip ()))
        Z_list.append (float (stripped_line[46:54].strip ()))

        atomtype = stripped_line[76:78].strip ()
        if atomtype == 'C':
            atomtype_list.append (1.0)  # 'h' means hydrophobic
        else:
            atomtype_list.append (-1.0)  # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


def reshape(X, Y, Z):
    X = np.reshape (np.array (X), (np.array (X).shape[0], 1))
    Y = np.reshape (np.array (Y), (np.array (Y).shape[0], 1))
    Z = np.reshape (np.array (Z), (np.array (Z).shape[0], 1))
    return X, Y, Z


def calculate_mean_indices(X, Y, Z):
    return sum (X) / len (X), sum (Y) / len (Y), sum (Z) / len (Z)


def normalize_indices(X, Y, Z, _X, _Y, _Z):
    X_norm = [a_i - _X for a_i in X]
    Y_norm = [a_i - _Y for a_i in Y]
    Z_norm = [a_i - _Z for a_i in Z]
    return X_norm, Y_norm, Z_norm