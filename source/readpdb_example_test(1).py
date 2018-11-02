import os

import numpy as np
from math import ceil, sin, cos, sqrt, pi
from source.network_3d import network
from keras.utils import to_categorical
from keras import optimizers, losses
import random

random.seed (11)

# path = "C:/Users/Dhananjaya/Dropbox/Deep Learning/Project/"
path = r"C:\Users\Sanjay Saha\CS5242-project\data20"


def read_pdb(filename):
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
        if line_length < 78:
            print ("ERROR: line length is different. Expected>=78, current={}".format (line_length))

        X_list.append (float (stripped_line[30:38].strip ()))
        Y_list.append (float (stripped_line[38:46].strip ()))
        Z_list.append (float (stripped_line[46:54].strip ()))

        atomtype = stripped_line[76:78].strip ()
        # Change this in both methods
        if atomtype == 'C':
            atomtype_list.append (1)  # ('h') # 'h' means hydrophobic TODO take this into account in betterway
        else:
            atomtype_list.append (2)  # ('p') # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


def read_test_pdb(filename):
    with open (filename, 'r') as file:
        strline_L = file.readlines ()
    # print(strline_L)

    X_list = list ()
    Y_list = list ()
    Z_list = list ()
    atomtype_list = list ()
    atomtype_list_ = list ()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip ()
        # print(stripped_line)

        splitted_line = stripped_line.split ('\t')

        X_list.append (float (splitted_line[0]))
        Y_list.append (float (splitted_line[1]))
        Z_list.append (float (splitted_line[2]))
        atomtype_list_.append (str (splitted_line[3]))

        # Change this in both methods
        if atomtype_list_ == 'h':
            atomtype_list.append (1)  # ('h') # 'h' means hydrophobic TODO take this into account in a betterway
        else:
            atomtype_list.append (2)  # ('p') # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


def make_grid(coords, features, grid_resolution=4.0, max_dist=96.0):
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
    f_shape = features.shape
    if len (f_shape) != 2 or f_shape[0] != N:
        raise ValueError ('features must be an array of floats of shape (%s, 3)'
                          % N)

    if not isinstance (grid_resolution, (float, int)):
        raise TypeError ('grid_resolution must be float')
    if grid_resolution <= 0:
        raise ValueError ('grid_resolution must be positive')

    if not isinstance (max_dist, (float, int)):
        raise TypeError ('max_dist must be float')
    if max_dist <= 0:
        raise ValueError ('max_dist must be positive')

    num_features = f_shape[1]
    max_dist = float (max_dist)
    grid_resolution = float (grid_resolution)

    box_size = ceil (2 * max_dist / grid_resolution + 1)

    # move all atoms to the neares grid point
    grid_coords = (coords + (max_dist / 2)) / grid_resolution
    grid_coords = grid_coords.round ().astype (int)

    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all (axis=1)
    grid = np.zeros ((box_size, box_size, box_size),
                     dtype=np.float32)
    for (x, y, z), f in zip (grid_coords[in_box], features[in_box]):
        grid[x, y, z] += f

    return grid


def calculate_mean_loc(X, Y, Z):
    return sum (X) / len (X), sum (Y) / len (Y), sum (Z) / len (Z)


def dataLoader(batch_size):
    L = 20
    # L = 2500
    # half of the batch(batch size/2) will have mapping pro ligand pairs and half will have non matching pairs
    # batch_size=batch_size/2
    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min (batch_end, L)
            train_x = []
            train_y = []
            print (batch_start)
            for i in range (batch_start, limit):
                pro_filename =  path + os.sep + "%.4d_pro_cg.pdb" % (i + 1)
                lig_filename = path + os.sep + "%.4d_lig_cg.pdb" % (i + 1)
                pro_X_list, pro_Y_list, pro_Z_list, pro_atomtype_list = read_pdb (pro_filename)
                lig_X_list, lig_Y_list, lig_Z_list, lig_atomtype_list = read_pdb (lig_filename)
                pro_coordinates = np.vstack (
                    (np.asarray (pro_X_list), np.asarray (pro_Y_list), np.asarray (pro_Z_list))).T
                pro_features = np.vstack ((np.asarray (pro_atomtype_list)))
                lig_coordinates = np.vstack (
                    (np.asarray (lig_X_list), np.asarray (lig_Y_list), np.asarray (lig_Z_list))).T
                lig_features = np.vstack ((np.asarray (lig_atomtype_list)))
                grid_p = make_grid (pro_coordinates, pro_features)  # 4,96)# 20, 480)
                grid_l = make_grid (lig_coordinates, lig_features)  # ,4,96)# 20, 480)
                grid_4d = []
                grid_4d.append (grid_p)
                grid_4d.append (grid_l)
                grid_4d = np.stack (grid_4d, 3)
                train_x.append (grid_4d)
                train_y.append (1)
            # create training data with not matching protene and ligand pairs
            for i in range (batch_start, limit):
                pro = random.randint (1, L)
                lig = random.randint (1, L)
                if pro != lig:
                    pro_filename = path + os.sep + "%.4d_pro_cg.pdb" % (pro)
                    lig_filename = path + os.sep + "%.4d_lig_cg.pdb" % (lig)
                else:
                    pro_filename = path + os.sep + "%.4d_pro_cg.pdb" % (pro)
                    lig_filename = path + os.sep + "%.4d_lig_cg.pdb" % (pro + 1)
                pro_X_list, pro_Y_list, pro_Z_list, pro_atomtype_list = read_pdb (pro_filename)
                lig_X_list, lig_Y_list, lig_Z_list, lig_atomtype_list = read_pdb (lig_filename)
                pro_coordinates = np.vstack (
                    (np.asarray (pro_X_list), np.asarray (pro_Y_list), np.asarray (pro_Z_list))).T
                pro_features = np.vstack ((np.asarray (pro_atomtype_list)))
                lig_coordinates = np.vstack (
                    (np.asarray (lig_X_list), np.asarray (lig_Y_list), np.asarray (lig_Z_list))).T
                lig_features = np.vstack ((np.asarray (lig_atomtype_list)))
                grid_p = make_grid (pro_coordinates, pro_features)  # 4,96)#20, 480)
                grid_l = make_grid (lig_coordinates, lig_features)  # , 4,96)#20, 480)
                grid_4d = [grid_p, grid_l]
                grid_4d = np.stack (grid_4d, 3)
                train_x.append (grid_4d)
                train_y.append (0)

            train_x = np.stack (train_x, 0)
            train_y = to_categorical (np.array (train_y))

            print (train_x.shape)
            yield (train_x, train_y)  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


# X_list, Y_list, Z_list, atomtype_list=read_pdb(path+"training_data/2060_pro_cg.pdb")
# X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_pro_cg.pdb")
# X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_lig_cg.pdb")
# print(X_list)
# print(Y_list)
# print(Z_list)
# print(atomtype_list)

pro_X_max_list = list ()
pro_Y_max_list = list ()
pro_Z_max_list = list ()
pro_X_min_list = list ()
pro_Y_min_list = list ()
pro_Z_min_list = list ()

train_x = [];
train_y = []
# for i in range (2500, 3000):
for i in range (1, 20):
    pro_filename = path + os.sep + "%.4d_pro_cg.pdb" % (i + 1)
    lig_filename = path + os.sep +  "%.4d_lig_cg.pdb" % (i + 1)
    pro_X_list, pro_Y_list, pro_Z_list, pro_atomtype_list = read_pdb (pro_filename)
    lig_X_list, lig_Y_list, lig_Z_list, lig_atomtype_list = read_pdb (lig_filename)
    pro_X_max_list.append (max (pro_X_list))
    pro_Y_max_list.append (max (pro_Y_list))
    pro_Z_max_list.append (max (pro_Z_list))
    pro_X_min_list.append (min (pro_X_list))
    pro_Y_min_list.append (min (pro_Y_list))
    pro_Z_min_list.append (min (pro_Z_list))
    # if(min(X_list) or min(Y_list) or min(Z_list)<0):
    #	print("negative values")
    pro_coordinates = np.vstack ((np.asarray (pro_X_list), np.asarray (pro_Y_list), np.asarray (pro_Z_list))).T
    pro_features = np.vstack ((np.asarray (pro_atomtype_list)))
    lig_coordinates = np.vstack ((np.asarray (lig_X_list), np.asarray (lig_Y_list), np.asarray (lig_Z_list))).T
    lig_features = np.vstack ((np.asarray (lig_atomtype_list)))
    grid_p = make_grid (pro_coordinates, pro_features)  # 4,96)#20, 480)
    grid_l = make_grid (lig_coordinates, lig_features)  # 4,96)#20, 480)
    grid_4d = []
    grid_4d.append (grid_p)
    grid_4d.append (grid_l)
    grid_4d = np.stack (grid_4d, 3)

    train_x.append (grid_4d)
    train_y.append (1)

# for i in range (2500, 3000):
for i in range (1, 20):
    # pro = random.randint (2500, 3000)
    pro = random.randint (1, 20)
    lig = random.randint (1, 20)
    if pro != lig:
        pro_filename = path + os.sep + "%.4d_pro_cg.pdb" % (pro)
        lig_filename = path + os.sep + "%.4d_lig_cg.pdb" % (lig)
    else:
        pro_filename = path + os.sep + "%.4d_pro_cg.pdb" % (pro)
        lig_filename = path + os.sep + "%.4d_lig_cg.pdb" % (pro + 1)
    pro_X_list, pro_Y_list, pro_Z_list, pro_atomtype_list = read_pdb (pro_filename)
    lig_X_list, lig_Y_list, lig_Z_list, lig_atomtype_list = read_pdb (lig_filename)
    pro_coordinates = np.vstack ((np.asarray (pro_X_list), np.asarray (pro_Y_list), np.asarray (pro_Z_list))).T
    pro_features = np.vstack ((np.asarray (pro_atomtype_list)))
    lig_coordinates = np.vstack ((np.asarray (lig_X_list), np.asarray (lig_Y_list), np.asarray (lig_Z_list))).T
    lig_features = np.vstack ((np.asarray (lig_atomtype_list)))
    grid_p = make_grid (pro_coordinates, pro_features)  # 20, 480)
    grid_l = make_grid (lig_coordinates, lig_features)  # , 20, 480)
    grid_4d = [grid_p, grid_l]
    grid_4d = np.stack (grid_4d, 3)
    train_x.append (grid_4d)
    train_y.append (0)

train_x = np.stack (train_x, 0)
train_y = to_categorical (np.array (train_y))


model = network (input_shape=(49, 49, 49, 2))
print (model.summary ())
sgd = optimizers.SGD (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile (sgd, 'categorical_crossentropy', metrics=['accuracy'])

# model.fit(x=train_x, y=train_y, epochs=5, verbose=1)
model.fit_generator (dataLoader (5), steps_per_epoch=500, epochs=1, verbose=1, callbacks=None,
                     validation_data=(train_x, train_y), validation_steps=None, class_weight=None, max_queue_size=10,
                     workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
# loss, acc = model.evaluate(x=train_x, y=train_y)
# steps_per_epoch was 500 epochs 20

# --------------------prediction stage -----------------------#
# print ("Predicting -----------------------------")
#
# filename = path + 'test_predictions.txt'
#
# f = open (filename, 'ab')
#
# train_x = [];
# train_y = []
# for i in range (1, 825):
#     pro_filename = path + "testing_data_release/testing_data/%.4d_pro_cg.pdb" % (i)
#     pro_X_list, pro_Y_list, pro_Z_list, pro_atomtype_list = read_test_pdb (pro_filename)
#     pro_coordinates = np.vstack ((np.asarray (pro_X_list), np.asarray (pro_Y_list), np.asarray (pro_Z_list))).T
#     pro_features = np.vstack ((np.asarray (pro_atomtype_list)))
#     grid_p = make_grid (pro_coordinates, pro_features)  # , 20, 480)
#     # calculate mean location of the protien
#     p_mean_X, p_mean_Y, p_mean_Z = calculate_mean_loc (pro_X_list, pro_Y_list, pro_Z_list)
#
#     test_x = [];
#     test_y = [];
#     test_y_array = []
#     for j in range (1, 825):
#         lig_filename = path + "testing_data_release/testing_data/%.4d_lig_cg.pdb" % (j)
#         lig_X_list, lig_Y_list, lig_Z_list, lig_atomtype_list = read_test_pdb (lig_filename)
#         # calculate mean location of the ligand
#         l_mean_X, l_mean_Y, l_mean_Z = calculate_mean_loc (lig_X_list, lig_Y_list, lig_Z_list)
#         # calculate difference mean distance between protein and ligand
#         mean_dif_X, mean_dif_Y, mean_dif_Z = abs (p_mean_X - l_mean_X), abs (p_mean_Y - l_mean_Y), abs (
#             p_mean_Z - l_mean_Z)
#
#         if (mean_dif_X < 15.0) & (mean_dif_Y < 15.0) & (mean_dif_Z < 15.0):
#             lig_coordinates = np.vstack ((np.asarray (lig_X_list), np.asarray (lig_Y_list), np.asarray (lig_Z_list))).T
#             lig_features = np.vstack ((np.asarray (lig_atomtype_list)))
#             grid_l = make_grid (lig_coordinates, lig_features)  # , 20, 480)
#             grid_4d = [grid_p, grid_l]
#             grid_4d = np.stack (grid_4d, 3)
#             grid_4d = np.expand_dims (grid_4d, axis=0)
#             # test_x.append(grid_4d)
#             # print(grid_4d.shape)
#             test_y = model.predict (grid_4d, batch_size=None, verbose=0, steps=None)
#             test_y_array.append (test_y[0][1])
#         # pritn('In')
#         else:
#             test_y_array.append (0)  # not matching
#         # print('Out')
#     print ('-------:prediction done for %d ' % i)
#     print (test_y_array)
#     np_test_y = np.asarray (test_y_array)
#     predictions = np.argpartition (np_test_y, -10)[-10:]
#     # print('predictions')
#     # print(predictions)
#     f.write (b'\r\n')
#     np.savetxt (f, (predictions), fmt='%d', delimiter='\t', newline=' ')
# f.close ()

'''
Average mean difference and std
8.380706433587736,8.130317374255398
9.07375417063099,8.500167262469123
9.638141614300395,9.325133731129021
'''
