"""
Created by Sanjay at 11/2/2018

Feature: Enter feature name here
Enter feature description here
"""

import os
import random
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix, accuracy_score

REPORT_DATA_PATH = r'C:\Users\Sanjay Saha\CS5242-project\report_data'


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype ('float') / cm.sum (axis=1)[:, np.newaxis]
        print ("Normalized confusion matrix")
    else:
        print ('Confusion matrix, without normalization')

    print (cm)

    plt.imshow (cm, interpolation='nearest', cmap=cmap)
    plt.title (title)
    plt.colorbar ()
    tick_marks = np.arange (len (classes))
    plt.xticks (tick_marks, classes, rotation=45)
    plt.yticks (tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max () / 2.
    for i, j in itertools.product (range (cm.shape[0]), range (cm.shape[1])):
        plt.text (j, i, format (cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.ylabel ('True label')
    plt.xlabel ('Predicted label')
    plt.tight_layout ()


def read_file(file):
    df = pd.read_csv (file, sep="\t", skiprows=[0, 1], header=None)
    df.drop (df.columns[0], axis=1, inplace=True)
    return df.values


if __name__ == '__main__':
    allfiles = [os.path.join (REPORT_DATA_PATH, f) for f in os.listdir (REPORT_DATA_PATH) if
                isfile (join (REPORT_DATA_PATH, f))]

    y_test = np.ones ((500,))  # Actual Test Values

    for file in allfiles:
        i = 0
        arr = read_file(file)
        y_pred = list ()
        for row in arr:
            i = i + 1  # Row number or, Protein File ID
            if i in row:
                y_pred.append (1)  # Match
            else:
                y_pred.append (0)  # Mismatch

        y_pred = np.array (y_pred)  # Predictions
        print ('Accuracy Score: ' + format (accuracy_score (y_test, y_pred)))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix (y_test, y_pred)
        np.set_printoptions (precision=2)

        # Plot non-normalized confusion matrix
        plt.figure ()
        plot_confusion_matrix (cnf_matrix, classes=['Match', 'Mismatch'],
                               title='Confusion matrix, without normalization')

