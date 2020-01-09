import numpy as np
import pandas as pd


def load_data(filepath):
    data = pd.read_csv('data.csv',header=None, sep=' ').values
    x = data[:, :-1]
    y = data[:, -1]

    return x, y

