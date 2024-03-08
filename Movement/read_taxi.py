import pandas as pd
import os
import random
import numpy as np

cnt = 0
candidate = []

for info in os.listdir('taxi'):
    domain = os.path.abspath('taxi')  
    file = os.path.join(domain, info)  
    data_ = pd.read_csv(file)
    data = data_.values
    flag = False
    for i in range(data.shape[0]):
        # in shanghai area
        if int(data[i][1].split(':')[-2]) <= 30:
            if (121.12 <= data[i][2] <= 121.72) \
                    and (30.9 <= data[i][3] <= 31.4):
                flag = True
        else:
            break

    if flag:
        # print(data[0][0])
        cnt += 1
        candidate.append(data[0][0])

print(candidate)

print('============================')
print(cnt)
print('============================')
print(random.sample(candidate, 10))

np.savetxt('sdv.txt', np.array(candidate), fmt='%i')

# selected = random.sample(candidate, 1000)
# np.savetxt('selected_sdv.txt', np.array(selected), fmt='%i')
