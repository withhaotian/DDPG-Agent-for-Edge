import os
import h5py

# from h5py
def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

def remove_incomplete_days(data, timestamps, T=48):
    days = [] # available days
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        # timestamps % 100 = current i_th day number
        if (timestamps[i] % 100) != 1:
            i += 1
        # this day has 48 timestamps
        elif i+T-1 < len(timestamps) and timestamps[i+T-1] % 100 == T:
            days.append(timestamps[i] // 100)
            i += T
        else:
            days_incomplete.append(timestamps[i] // 100)
            i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if (t // 100) in days:
                idx.append(i) # numbers of valid data

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        return data, timestamps