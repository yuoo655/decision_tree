import numpy as np
from load_data import load_data
from cross_validation import *

if __name__ == "__main__":
    x, y = load_data('./data.csv')
    average_accuracy = k_folds_cross_validation(x, y, 5, 'c4.5')
    average_accuracy = k_folds_cross_validation(x, y, 6, 'cart')