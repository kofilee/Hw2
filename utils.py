import os
import numpy as np
from pickle import dump
from pickle import load
from sklearn.preprocessing import StandardScaler


def read_data(number, test):
    fault_number = "%02d" % number
    if test:
        path = 'd' + fault_number + '_te'
    else:
        path = 'd' + fault_number
    path = os.getcwd() + '/data/' + path + ".dat"
    data = np.loadtxt(path)
    if not (number or test):
        data = data.T
    else:
        data = np.delete(data, list(range(0, 20)), axis=0)
    data = standardization(data, number + test)
    return data


def standardization(data, fit):
    scalar = StandardScaler()
    if fit == 0:
        scalar.fit(data)
        dump(scalar, open('scalar.pkl', 'wb'))
    else:
        scalar = load(open('scalar.pkl', 'rb'))
    stand_data = scalar.transform(data)
    return stand_data

