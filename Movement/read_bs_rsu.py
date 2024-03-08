import pandas as pd
import random
import numpy as np

rsu = []
candidate_rsu = []
for i in range(30):
    candidate_rsu.append([])

candidate_bs = []

data_1 = pd.read_csv('telecom_data/data_7.1~7.15.csv').values
data_2 = pd.read_csv('telecom_data/data_7.16~7.31.csv').values

for i in range(data_1.shape[0]):
    loc = [data_1[i][5], data_1[i][4]]
    idx = int((data_1[i][5]-121.12)/0.1) + int((data_1[i][4] - 30.9)/0.1)*6
    if loc not in candidate_rsu[idx]:
        candidate_rsu[idx].append(loc)
        rsu.append(loc)

for i in range(data_1.shape[0]):
    loc = [data_2[i][5], data_2[i][4]]
    idx = int((data_2[i][5]-121.12)/0.1) + int((data_2[i][4] - 30.9)/0.1)*6
    if loc not in candidate_rsu[idx]:
        candidate_rsu[idx].append(loc)
        rsu.append(loc)

# get the final rsu selection
ret = []
for i in range(30):
    ret.append(random.sample(candidate_rsu[i], 1))

# print(ret)
#
# selested_loc_bs = [7, 10, 19, 22] # selected bs location idx
#
# for i in selested_loc_bs:
#     a = random.sample(candidate_rsu[i], 3)[-1]
#     while a in ret:
#         a = random.sample(candidate_rsu[i], 3)[-1]
#     candidate_bs.append(a)
#
# print(candidate_bs)

np.savetxt('rsu.txt', np.array(rsu).reshape(len(rsu), 2), fmt='%.06f')
# np.savetxt('selected_bs.txt', np.array(candidate_bs).reshape(4, 2), fmt='%.06f')
