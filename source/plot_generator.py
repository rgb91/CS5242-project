"""
Created by Sanjay at 11/2/2018

Feature: Enter feature name here
Enter feature description here
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

READ_PATH = r'C:\Users\Sanjay Saha\CS5242-project\report_data\acc_loss'
SAVE_PATH = r'C:\Users\Sanjay Saha\CS5242-project\report_data\saved_plots'


def get_arrays_from_file(file):
    df = pd.read_csv (file, sep=",", header=None)
    df_np = df.values
    return df_np[0, :], df_np[1, :], df_np[2, :], df_np[3, :]


def plot_arrays(save_file_name):
    # Plot and save Acc & Loss vs Epoch
    plt.figure ()
    plt.plot (np.arange (1, len (train_acc) + 1), train_acc, c='green', label='Train Acc')
    plt.plot (np.arange (1, len (train_acc) + 1), val_acc, c='blue', label='Val Acc')
    plt.plot (np.arange (1, len (train_acc) + 1), train_loss, c='orange', label='Train Loss')
    plt.plot (np.arange (1, len (train_acc) + 1), val_loss, c='lightblue', label='Val Loss')
    plt.title ('Train, Validation - Acc, Loss')
    plt.xlabel ('Epochs')
    plt.ylabel ('Acc and Loss')
    plt.legend ()
    plt.grid ()
    plt.savefig (os.path.join(SAVE_PATH, save_file_name))
    plt.show ()
    plt.clf()


if __name__ == '__main__':
    allfiles = [f for f in os.listdir (READ_PATH) if
                os.path.isfile (os.path.join (READ_PATH, f))]
    for file in allfiles:
        train_acc, val_acc, train_loss, val_loss = get_arrays_from_file (os.path.join (READ_PATH, file))
        file_name, file_extension = os.path.splitext(file)
        plot_arrays(file_name)
