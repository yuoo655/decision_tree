import numpy as np


def load_data(filepath):

    with open(filepath, 'r') as f:
        raw_data = f.readlines()

    raw_data_lable  = np.zeros(len(raw_data),dtype=np.int16)           # 0 means M = malignant,   1 means B = benign

    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i].split(' ')

        raw_data[i][30] = int(raw_data[i][30])
        raw_data_lable[i] = int(raw_data[i][30])
        # raw_data[i] = raw_data[i][:31]

    raw_data = np.array(raw_data, dtype=float)
    raw_data = raw_data[:,:30]
    return raw_data, raw_data_lable




# def load_data(filepath):


#     with open(filepath, 'r') as f:
#         raw_data = f.readlines()

#     raw_data_lable  = np.zeros(len(raw_data),dtype=np.int16)           # 0 means M = malignant,   1 means B = benign

#     for i in range(len(raw_data)):
#         raw_data[i] = raw_data[i].split(' ')
#         raw_data[i][30] = int(raw_data[i][30])
#         raw_data_lable[i] = int(raw_data[i][30])
#         raw_data[i] = raw_data[i][:30]

#     raw_data = np.array(raw_data, dtype=float)


#     return raw_data, raw_data_lable
