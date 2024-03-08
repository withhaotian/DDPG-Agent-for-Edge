import h5py
import numpy as np
import os, sys
from copy import copy
from .STMatrix import ST_Matrix
import time
from datetime import datetime, date
DATAPATH = os.path.dirname(__file__)  

class MinMaxNormalization(object):
    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print('min: ', self._min, 'max: ', self._max)

    def transform(self, X):
        # if self._max - self._min == 0:
        #     X = X + 0.1
        X = 1. * (X - self._min)/(self._max - self._min)
        print("Transform is done!")
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X

def timestamp2vec(timestamps):
    print(timestamps[0])
    date = [time.strptime(str(t)[:8], '%Y%m%d').tm_wday for t in timestamps]
    ret = []
    for i in date:
        # one-hot label
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)
        else:
            v.append(1)
        ret.append(v)
    return np.asarray(ret)


def load_data(filename, poi_data, T=48, len_clossness=None, len_period=None, len_trend=None,
              len_test=None, is_poi_time=False, evaluate=False, type=None):
    assert(not(len_clossness is None and len_period is None and len_trend is None))

    if not evaluate:
        f = h5py.File(f'{filename}', 'r')
    else:
        f = h5py.File(f'{filename}'.format(type), 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()

    # data, timestamps = remove_incomplete_days(data, timestamps, T)
    data[data < 0] = 0.

    data_all = [data]
    timestamps_all = [timestamps]

    data_train = data[:len(data)-len_test]

    # use data_train to instantiate min_max_scale
    mmn = MinMaxNormalization()
    mmn.fit(data_train)

    # mmn = MM(np.max(data), np.min(data))

    # MinMax Normalization
    data_all_mmn = [mmn.transform(d) for d in data_all]


    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []

    for data, timestamps in zip(data_all_mmn, timestamps_all):
        st = ST_Matrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(len_clossness=len_clossness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)

    XC_train, XP_train, XT_train, Y_train = \
        XC[:len(XC)-len_test], XP[:len(XP)-len_test], XT[:len(XT)-len_test], Y[:len(Y)-len_test]

    XC_test, XP_test, XT_test, Y_test = \
        XC[len(XC)-len_test:], XP[len(XP)-len_test:], XT[len(XT)-len_test:], Y[len(Y)-len_test:]

    timestamp_train, timestamp_test = \
        timestamps_Y[:len(timestamps_Y)-len_test], timestamps[len(timestamps_Y)-len_test:]

    X_train = []
    X_test = []

    for l, _X in zip([len_clossness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(_X)
    for l, _X in zip([len_clossness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(_X)

    # DSTN+resplus+poi+time
    if is_poi_time:
        poi_x = np.load(f'{poi_data}')
        poi = poi_x[:]
        len_train = XC_train.shape[0]
        map_height = data.shape[1]
        map_width = data.shape[2]
        for i in range(poi.shape[0]):
            poi[i] = poi[i] / np.max(poi[i])

        P_train = np.repeat(poi.reshape(1, poi.shape[0], map_height, map_width), len_train, axis=0)
        P_test = np.repeat(poi.reshape(1, poi.shape[0], map_height, map_width), len_test, axis=0)

        len_total = data.shape[0]
        T_period = 24

        # for time
        time = np.arange(len_total, dtype=int)
        # hour
        time_hour = time % T_period
        matrix_hour = np.zeros([len_total, 48, map_height, map_width])
        for i in range(len_total):
            matrix_hour[i, time_hour[i], :, :] = 1
        # day
        time_day = (time // T_period) % 7
        matrix_day = np.zeros([len_total, 7, map_height, map_width])
        for i in range(len_total):
            matrix_day[i, time_day[i], :, :] = 1
        # con
        matrix_T = np.concatenate((matrix_hour, matrix_day), axis=1)

        matrix_T = matrix_T[:len(XC)]

        T_train = matrix_T[:len(matrix_T)-len_test]
        T_test = matrix_T[len(matrix_T)-len_test:]

    else:
        P_train = []
        P_test = []
        T_train = []
        T_test = []

    return X_train, Y_train, X_test, Y_test, mmn, P_train, P_test, T_train, T_test


