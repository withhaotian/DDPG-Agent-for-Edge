import os
import pandas as pd
import numpy as np
from datetime import datetime

def string2timestamps(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T 
    num_per_T = T // 24 
    for t in strings:
        year, month, day, slot = int(str(t)[:4]), int(str(t)[4:6]), int(str(t)[6:8]), int(str(t)[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60 * time_per_slot))))

    return timestamps

class ST_Matrix(object):
    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(ST_Matrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamps(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0
   
    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True
    def create_dataset(self, len_clossness=3, len_trend=3, TrendInterval=2, len_period=3, PeriodInterval=1):
        offset_frame = pd.DateOffset(minutes=24*60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_clossness+1),
                   [int(PeriodInterval * self.T * j) for j in range(1, len_period+1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]]
        
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_clossness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = [self.get_matrix(self.pd_timestamps[i])]
            
            if len_clossness > 0:
                XC.append(np.stack(x_c))
            if len_period > 0:
                XP.append(np.stack(x_p))
            if len_trend > 0:
                XT.append(np.stack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        return XC, XP, XT, Y, timestamps_Y