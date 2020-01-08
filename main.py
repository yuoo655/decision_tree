import numpy as np
from load_data import load_data
from classifier_c45 import *
from cross_validation import *

# x, y = load_data('./data.csv')
# average_accuracy = k_folds_cross_validation(x, y, 10, 'c4.5')

if __name__ == "__main__":
    x, y = load_data('./data.csv')
    average_accuracy = k_folds_cross_validation(x, y, 10, 'c4.5')