#####################  configuration  ####################
b = 1
Kb = 1024
Mb = 1024 * Kb
Gb = 1024 * Mb
Hz = 1
KHz = 1000 * Hz
MHz = 1000 * KHz
GHz = 1000 * MHz
cyc = 1
Kcyc = 1000 * cyc
Mcyc = 1000 * Kcyc
Gcyc = 1000 * Mcyc
############################################################

SDV_NUM = 70
RSU_NUM = 30
BS_NUM = 4
EDGE_NUM = RSU_NUM+BS_NUM
CONTENT_TYPE_NUM = 10

LIMIT = 3

B_BOUND = 20 * MHz
CACH_CAP_BS = 800 * Mb * 8
CACH_CAP_RSU = 500 * Mb * 8
R_BOUND = 1 * GHz

# 60,70,80,90,100 MB: Toward Energy-Aware Caching for Intelligent Connected Vehicles
req_u2e_size = 100 * Mb * 8
# 10,30,50,70,90 M: Imitation Learning Enabled Task Scheduling for Online Vehicular Edge Computing
process_loading = 90 * Mcyc


print('===========SDV_NUM:', SDV_NUM, '=============')
print('===========EDGE_NUM:', RSU_NUM, '=============')
print('===========BS_NUM:', BS_NUM, '=============')
print('===========B_BOUND:', B_BOUND, '=============')
print('===========CACHE_RSU:', CACH_CAP_RSU/8, '=============')
print('===========REQ_SIZE:', req_u2e_size/8, '=============')
print('===========PROCESSING:', process_loading, '=============')