import numpy as np
import pandas as pd
import random
#
# '''ground-truth with 6 days'''
# all_per = np.zeros((10, 5, 6))      # ground-truth of 10 types of file
# all_data = np.zeros((5, 6))         # all ground-truths
# for i in range(10):
#     data = np.load('y_label_{}.npy'.format(i+1))
#     data_ = data.reshape(data.shape[0], data.shape[2], data.shape[3])
#     # data_ = data_[:, 4:10, 7:13]
#     # print(data_)
#     # for j in range(data_.shape[0]):
#     for k in range(data_.shape[1]):
#         for l in range(data_.shape[2]):
#             all_per[i][k][l] += data_[17][k][l]
#             all_data[k][l] += data_[17][k][l]
# print(all_per)
# print(all_data)
# print('===============================')
#
# '''predicted rsults with 6 days'''
# pred_per = np.zeros((10, 5, 6))     # prediction of 10 types of file
# pred_data = np.zeros((5, 6))        # all prediction results
# for i in range(10):
#     data = np.load('y_pred_{}.npy'.format(i + 1))
#     data_ = data.reshape(data.shape[0], data.shape[2], data.shape[3])
#     # for j in range(data_.shape[0]):
#     for k in range(data_.shape[1]):
#         for l in range(data_.shape[2]):
#             pred_per[i][k][l] += data_[17][k][l]
#             pred_data[k][l] += data_[17][k][l]
#
# print(pred_per)
# print(pred_data)
# print('===============================')
#
# '''popularity prediction results with 6 days'''
# pred_pop = np.zeros((10, 5, 6))
# avg_pred_pop = np.zeros(10)
#
# label_pop = np.zeros((10, 5, 6))
# avg_label_pop = np.zeros(10)
#
# for i in range(10):
#     for j in range(pred_per.shape[1]):
#         for k in range(pred_per.shape[2]):
#             pred_pop[i][j][k] = pred_per[i][j][k] / pred_data[j][k]
#             label_pop[i][j][k] = all_per[i][j][k] / all_data[j][k]
#             avg_pred_pop[i] += pred_pop[i][j][k]
#             avg_label_pop[i] += label_pop[i][j][k]
#
# avg_label_pop /= 30
# avg_pred_pop /= 30
#
# print(label_pop)
# print(pred_pop)
# print('===============================')
# print(avg_label_pop)
# print(avg_pred_pop)

# '''saving results'''
# # save = pd.DataFrame(avg_label_pop)
# # save.to_csv('ground-truth popularity.csv')
#
# save = pd.DataFrame(avg_pred_pop)
# save.to_csv('predicted popularity.csv')

"""----------------------------------------------------------------------------------------"""

'''ground-truth with 6 days'''
all_per = np.zeros(10)      # ground-truth of 10 types of file
all_data = 0        # all ground-truths
for i in range(10):
    data = np.load('y_label_{}.npy'.format(i+1))
    data_ = data.reshape(data.shape[0], data.shape[2], data.shape[3])
    # for j in range(data_.shape[0]):
    for k in range(data_.shape[1]):
        for l in range(data_.shape[2]):
            all_per[i] += data_[113][k][l]
            all_data += data_[113][k][l]
print(all_per)
print(all_data)
print('===============================')

'''predicted rsults with 6 days'''
pred_per = np.zeros(10)     # prediction of 10 types of file
pred_data = 0        # all prediction results
for i in range(10):
    data = np.load('y_pred_{}.npy'.format(i + 1))
    data_ = data.reshape(data.shape[0], data.shape[2], data.shape[3])
    # for j in range(data_.shape[0]):
    for k in range(data_.shape[1]):
        for l in range(data_.shape[2]):
            pred_per[i] += data_[113][k][l]
            pred_data += data_[113][k][l]

print(pred_per)
print(pred_data)
print('===============================')

'''popularity prediction results with 6 days'''
pred_pop = np.zeros(10)
label_pop = np.zeros(10)

for i in range(10):
    pred_pop[i] = pred_per[i] / pred_data
    label_pop[i] = all_per[i] / all_data

print(label_pop)
print(pred_pop)

'''saving results'''
save = pd.DataFrame(label_pop)
save.to_csv('ground-truth popularity.csv')

save = pd.DataFrame(pred_pop)
save.to_csv('predicted popularity.csv')

np.savetxt('ground-truth popularity.txt', label_pop, fmt='%.6f')
np.savetxt('predicted popularity.txt', pred_pop, fmt='%.6f')