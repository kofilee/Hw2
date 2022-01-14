import numpy as np
from utils import read_data
from tensorflow import one_hot


def data_loader(timesteps, type_list, te=0):
    num_type = len(type_list)
    data_all = []
    label_all = []

    for idx, fault_type_num in enumerate(type_list):

        data = read_data(fault_type_num, te)
        no, dim = data.shape
        batch_size = no - timesteps + 1

        windows = np.empty((0, timesteps, dim))
        for j in range(batch_size):
            windows = np.vstack((data[np.newaxis, j:j + timesteps, ], windows))

        label = one_hot([idx] * batch_size, num_type)

        if len(data_all):
            data_all = np.vstack((data_all, windows))
            label_all = np.vstack((label_all, label))
        else:
            data_all = windows
            label_all = label

    return data_all, label_all
