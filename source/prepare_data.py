"""
Created by Sanjay at 10/5/2018

Feature: Prepare PDB Files for
Training the CNN Model
"""
import os
import random
import numpy as np
from os import listdir
from os.path import isfile, join
from math import ceil, sin, cos, sqrt, pi
from source.utils import reshape, calculate_mean_indices, normalize_indices, read_pdb_file
from itertools import combinations

DATASET_DIRECTORY_PATH = r'C:\Users\Sanjay Saha\CS5242-project\data20'
DATASET_OUTPUT_PATH = r'C:\Users\Sanjay Saha\CS5242-project\processed_train_data'
MAX_DIST = 50.0  # 10 means box size is 21 if grid resolution is 1.0
RESOLUTION = 2.0


def make_grid(coords, features, grid_resolution=RESOLUTION, max_dist=10.0):
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.

    Parameters
    ----------
    coords, features: array-likes, shape (N, 3) and (N, F)
        Arrays with coordinates and features for each atoms.
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.

    Returns
    -------
    coords: np.ndarray, shape = (M, M, M, F)
        4D array with atom properties distributed in 3D space. M is equal to
        2 * `max_dist` / `grid_resolution` + 1
    """

    try:
        coords = np.asarray (coords, dtype=np.float)
    except ValueError:
        raise ValueError ('coords must be an array of floats of shape (N, 3)')
    c_shape = coords.shape
    if len (c_shape) != 2 or c_shape[1] != 3:
        raise ValueError ('coords must be an array of floats of shape (N, 3)')

    N = len (coords)
    try:
        features = np.asarray (features, dtype=np.float)
    except ValueError:
        raise ValueError ('features must be an array of floats of shape (N, 3)')
    if not isinstance (grid_resolution, (float, int)):
        raise TypeError ('grid_resolution must be float')
    if grid_resolution <= 0:
        raise ValueError ('grid_resolution must be positive')

    if not isinstance (max_dist, (float, int)):
        raise TypeError ('max_dist must be float')
    f_shape = features.shape
    if len (f_shape) != 2 or f_shape[0] != N:
        raise ValueError ('features must be an array of floats of shape (%s, 3)' % N)

    if max_dist <= 0:
        raise ValueError ('max_dist must be positive')

    num_features = f_shape[1]
    max_dist = float (max_dist)
    grid_resolution = float (grid_resolution)

    box_size = ceil (2 * max_dist / grid_resolution + 1)

    # move all atoms to the neares grid point
    # grid_coords = (coords + max_dist) / grid_resolution # previous calculation
    grid_coords = (coords) / grid_resolution
    grid_coords = grid_coords.round ().astype (int)

    # remove atoms outside the box
    # in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all (axis=1) # original
    highest_bound = int (round (sqrt (3) / 2 * box_size))
    lowest_bound = int (round (-highest_bound))
    in_box = ((grid_coords >= lowest_bound) & (grid_coords < highest_bound)).all (axis=1)
    grid = np.zeros ((1, box_size, box_size, box_size, num_features),
                     dtype=np.float32)
    for (x, y, z), f in zip (grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f

    return grid


def check_grid(grid):
    """
    Prints the number of protein and ligand atoms are there in the grid.
    :param grid: (N x N x N x F)
    :return: None
    """
    temp_grid = np.reshape (grid, (grid.shape[1], grid.shape[2], grid.shape[3], grid.shape[4]))
    # print('Temp grid shape: '+str(temp_grid.shape))
    box_size = grid.shape[1]
    lig_count, pro_count, blank_count = 0, 0, 0
    for i in range (box_size):
        for j in range (box_size):
            for k in range (box_size):
                if temp_grid[i, j, k, 1] == -1:
                    lig_count += 1
                elif temp_grid[i, j, k, 1] == 1:
                    pro_count += 1
                elif temp_grid[i, j, k, 1] == 0:
                    blank_count += 1
    print ('Protein Atoms in Box: ' + str (pro_count))
    print ('Ligand Atoms in Box: ' + str (lig_count))
    print ('Blank spaces in Box: ' + str (blank_count))
    print ('Shape of grid: ' + str (grid.shape))


def get_grid_from_file_pairs(ligandfile, proteinfile):
    # Read protein and ligand coordinates and type from PDB files
    X_list_ligand, Y_list_ligand, Z_list_ligand, atomtype_list_ligand = read_pdb_file (ligandfile)
    X_list_protein, Y_list_protein, Z_list_protein, atomtype_list_protein = read_pdb_file (proteinfile)

    # Normalize the coordinates: Put the ligands in the middle and align around the mean of ligand atoms
    X_mean_ligand, Y_mean_ligand, Z_mean_ligand = calculate_mean_indices (X_list_ligand, Y_list_ligand,
                                                                          Z_list_ligand)
    X_list_ligand, Y_list_ligand, Z_list_ligand = normalize_indices (X_list_ligand, Y_list_ligand, Z_list_ligand,
                                                                     X_mean_ligand, Y_mean_ligand, Z_mean_ligand)
    X_list_protein, Y_list_protein, Z_list_protein = normalize_indices (X_list_protein, Y_list_protein,
                                                                        Z_list_protein,
                                                                        X_mean_ligand, Y_mean_ligand, Z_mean_ligand)

    # Reshape
    X_list_ligand, Y_list_ligand, Z_list_ligand = reshape (X_list_ligand, Y_list_ligand, Z_list_ligand)
    X_list_protein, Y_list_protein, Z_list_protein = reshape (X_list_protein, Y_list_protein, Z_list_protein)

    # Concatenating proteins and ligands' coordinates
    X_list = np.concatenate ((X_list_protein, X_list_ligand))
    Y_list = np.concatenate ((Y_list_protein, Y_list_ligand))
    Z_list = np.concatenate ((Z_list_protein, Z_list_ligand))

    # Preparing atom type list
    atomtype_list_ligand = np.array (atomtype_list_ligand)
    atomtype_list_ligand = np.reshape (atomtype_list_ligand, (atomtype_list_ligand.shape[0], 1))
    atomtype_list_protein = np.array (atomtype_list_protein)
    atomtype_list_protein = np.reshape (atomtype_list_protein, (atomtype_list_protein.shape[0], 1))
    atomtype_list_pair = np.asarray (np.concatenate ((atomtype_list_ligand, atomtype_list_protein)), dtype='float')

    # Marking atoms as protein and ligand
    protein_identifier = np.ones ((atomtype_list_protein.shape[0], 1), dtype='float')
    ligand_identifier = np.negative (np.ones ((atomtype_list_ligand.shape[0], 1), dtype='float'))
    protein_ligand_identifier = np.concatenate ((protein_identifier, ligand_identifier))

    # Feature Channel (N x 2) containing - 1) Atom type & 2) If an atom is protein or ligand
    feature_channel = np.concatenate ((atomtype_list_pair, protein_ligand_identifier), axis=1)

    coordinates_one_pair = np.asarray (np.concatenate ((X_list, Y_list, Z_list), axis=1), dtype='float')

    # Create the GRID
    grid = make_grid (coordinates_one_pair, feature_channel, max_dist=MAX_DIST)

    # Check grid
    # print ()
    # print ('Total Protein Atoms: ' + str (atomtype_list_protein.shape[0]))
    # print ('Total Ligand Atoms: ' + str (atomtype_list_ligand.shape[0]))
    # check_grid (grid)
    # print ('====================================================')

    return grid


if __name__ == '__main__':
    datapath = DATASET_DIRECTORY_PATH
    allfiles = [join (datapath, f) for f in listdir (datapath) if isfile (join (datapath, f))]
    X = []
    y = []

    # Prepare positive examples of dataset
    for i in range (0, len (allfiles), 2):
        grid = get_grid_from_file_pairs (ligandfile=allfiles[i], proteinfile=allfiles[i + 1])
        X.append (grid)
        y.append (1.0)

    # Prepare negative examples of dataset
    for i in range (0, len (allfiles), 2):
        random_index = random.randint (0, int (len (allfiles) / 2))
        while random_index == i:
            random_index = random.randint (0, int (len (allfiles) / 2))
        if random_index % 2 == 0:
            random_index += 1
        grid = get_grid_from_file_pairs (ligandfile=allfiles[i], proteinfile=allfiles[random_index])
        X.append (grid)
        y.append (0.0)

    m = len (X)
    X = np.array (X)
    X = np.reshape (X, (X.shape[0], X.shape[2], X.shape[3], X.shape[4], X.shape[5]))
    y = np.array (y)
    y = np.reshape (y, (y.shape[0], 1))
    print (X.shape)
    print (y.shape)
    # dataset = np.concatenate((X, y.T), axis=1)
    np.save (os.path.join (DATASET_OUTPUT_PATH, 'X_val.npy'), X)
    np.save (os.path.join (DATASET_OUTPUT_PATH, 'y_val.npy'), y)
