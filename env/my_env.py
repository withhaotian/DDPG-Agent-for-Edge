import random
import numpy as np
import math
import pandas as pd
import os
from utils import configs

random.seed(42)     # for reproducibility

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

#####################  hyper parameters  ####################
SDV_NUM = configs.SDV_NUM
RSU_NUM = configs.RSU_NUM
BS_NUM = configs.BS_NUM
EDGE_NUM = RSU_NUM+BS_NUM
CONTENT_TYPE_NUM = configs.CONTENT_TYPE_NUM
LIMIT = configs.LIMIT
B_BOUND = configs.B_BOUND
CACH_CAP_BS = configs.CACH_CAP_BS
CACH_CAP_RSU = configs.CACH_CAP_RSU
R_BOUND = configs.R_BOUND
req_u2e_size = configs.req_u2e_size
process_loading = configs.process_loading
########################################################

#####################  functions  ####################
def geodistance(lng1, lat1, lng2, lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(math.radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # transformation
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance = 2 * math.asin(math.sqrt(a)) * 6371 * 1000  # 6371km
    distance = round(distance / 1000, 3)
    return distance  # km

# print(geodistance(121.12, 31.4, 121.72, 31.4), ' KM')

def trans_rate(user_loc, edge_loc, band):
    k = 0.01  # path_loss
    e = 5  # exponent
    # d = np.sqrt(np.sum(np.square(user_loc[0] - edge_loc)))  # distance
    d = geodistance(user_loc[0], user_loc[1], edge_loc[0], edge_loc[1])  # km
    gbu = k * math.pow(d, -e)
    pb = 40  # transmission (W)
    temp = -174  # gauss white noise power (dBm/HZ)
    o = math.pow(10, temp / 10) * 0.001  # W/HZ
    SNR = pb * gbu / o
    B = band  # HZ
    w = B * (math.log(1 + SNR) / math.log(2))
    return w    # bps

# print(50*Mb / trans_rate((121.254572, 31.390234), (120.618727, 30.442177), b_bound), ' sec')

def BandwidthTable(edge_num):
    BandwidthTable = np.zeros((edge_num, edge_num))
    for i in range(0, edge_num):
        for j in range(i + 1, edge_num):
            if i >= RSU_NUM or j >= RSU_NUM:
                BandwidthTable[i][j] = B_BOUND * 2      # bs
            else:
                BandwidthTable[i][j] = B_BOUND          # rsu
    return BandwidthTable

def two_to_one(two_table):
    one_table = two_table.flatten()
    return one_table

def p_table():
    return pd.read_csv('utils/predicted popularity.txt', header=None).values

def generate_state(two_table, U, E):
    # initial
    one_table = two_to_one(two_table)
    pop = p_table()
    S = np.zeros(len(U) + len(U) + CONTENT_TYPE_NUM + len(E) + one_table.size + len(U) + len(U) * 2)
    # print("one_table.size=",one_table.size)
    count = 0
    # requsted resources
    for user in U:
        S[count] = user.req.tasktype.process_loading / Mcyc
        count += 1
    # requested size of data
    for user in U:
        S[count] = user.req.tasktype.req_u2e_size / Mb
        count += 1
    # popularity of task
    for i in range(CONTENT_TYPE_NUM):
        S[count] = pop[i] * 100
        count += 1
    # available resource of each edge server
    for edge in E:
        S[count] = edge.capability / GHz
        count += 1
    # available bandwidth of each connection
    for i in range(len(one_table)):
        S[count] = one_table[i] / MHz
        count += 1
    # location of the user
    for user in U:
        S[count] = user.loc[0]
        S[count + 1] = user.loc[1]
        count += 2
    return S

# get rsu location
def proper_edge_loc(edge_num):
    locs = pd.read_csv('Movement/selected_rsu.txt', header=None, sep=' ').values.tolist()
    sel_locs = random.sample(locs, edge_num)
    return sel_locs     # return the EDGE_NUM rsu locations

def proper_bs_loc():
    locs = pd.read_csv('Movement/selected_bs.txt', header=None, sep=' ').values.tolist()
    return locs

#############################  UE  ###########################
class UE():
    def __init__(self, user_id, data_idx):
        self.user_id = user_id  # number of the user
        self.num_step = 0  # the number of step
        # print(data_idx)
        self.data_idx = data_idx
        self.mob = self.read_movements(self.data_idx)
        # print(self.mob)
        self.loc = [self.mob[0][2], self.mob[0][3]]     # initiliazed location
        self.loc_idx = 0
        # print(self.loc)

    # load the taxi datasets
    def read_movements(self, mob_idx):
        for info in os.listdir('Movement/taxi'):
            if info == f'Taxi_{mob_idx}':
                domain = os.path.abspath('Movement/taxi')
                file = os.path.join(domain, info)
                data = pd.read_csv(file, header=None, usecols=[0, 1, 2, 3], skiprows=lambda x: x > 100).values       # use vehicle id, date, longi, lati
                return data

    # specific probability of requested task type
    def number_of_certain_probability(self, sequence, probability):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item

    def generate_request(self, edge_id, cach_list):
        value_list = [i + 1 for i in range(CONTENT_TYPE_NUM)]
        real_pop_ = pd.read_csv('utils/ground-truth popularity.txt', header=None).values
        real_pop = []
        for i in range(CONTENT_TYPE_NUM):
            real_pop.append(real_pop_[i][0])
        # print('================================')
        # print(real_pop.tolist())
        self.req = Request(self.user_id, edge_id, self.number_of_certain_probability(value_list, real_pop),
                        cach_list)

    def request_update(self, b):
        # default request.state == 5 means disconnection
        # 6 means migration (not implement)
        if self.req.state == 5:
            self.req.timer += 1
        else:
            self.req.timer = 0
            if self.req.state == 0:
                self.req.state = 1
                self.req.u2e_size = self.req.tasktype.req_u2e_size
                if self.req.u2e_size > 0:
                    if self.req.u2e_size > trans_rate(self.loc, self.req.edge_loc, b):
                        self.req.delay += 1
                    else:
                        self.req.delay += self.req.u2e_size / trans_rate(self.loc, self.req.edge_loc, b)
                    self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc, b)
            elif self.req.state == 1:
                if self.req.u2e_size > 0:
                    if self.req.u2e_size > trans_rate(self.loc, self.req.edge_loc, b):
                        self.req.delay += 1
                    else:
                        self.req.delay += self.req.u2e_size / trans_rate(self.loc, self.req.edge_loc, b)
                    self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc, b)
                else:
                    self.req.state = 2
                    self.req.process_size = self.req.tasktype.process_loading
                    if self.req.process_size > 0:
                        if self.req.process_size > self.req.resource:
                            self.req.delay += 1
                        else:
                            self.req.delay += self.req.process_size / self.req.resource
                        # print("process_size:{} | capacity:{}".format(self.req.process_size,
                        #     self.req.resource))
                        self.req.process_size -= self.req.resource
            elif self.req.state == 2:
                if self.req.process_size > 0:
                    if self.req.process_size > self.req.resource:
                        self.req.delay += 1
                    else:
                        self.req.delay += self.req.process_size / self.req.resource
                    # print("self.req.process_size resource", self.req.process_size,
                    #       self.req.resource)
                    self.req.process_size -= self.req.resource
                else:
                    self.req.state = 3
                    self.req.e2u_size = self.req.tasktype.req_e2u_size
                    if self.req.e2u_size > trans_rate(self.loc, self.req.edge_loc, b):
                        self.req.delay += 1
                    else:
                        self.req.delay += self.req.e2u_size / trans_rate(self.loc, self.req.edge_loc, b)
                    self.req.e2u_size -= trans_rate(self.loc, self.req.edge_loc, b)  # value is small,so simplify
            else:
                if self.req.e2u_size > 0:
                    if self.req.e2u_size > trans_rate(self.loc, self.req.edge_loc, b):
                        self.req.delay += 1
                    else:
                        self.req.delay += self.req.e2u_size / trans_rate(self.loc, self.req.edge_loc, b)
                    self.req.e2u_size -= trans_rate(self.loc, self.req.edge_loc, b)
                else:
                    self.req.state = 4
        # print("transmission rate:", trans_rate(self.loc, self.req.edge_loc))

    def mobility_update(self, time):  # t: second
        # the time for moving
        cum_sec = int(self.mob[self.loc_idx+1][1].split(':')[-2])*60 \
                  + int(self.mob[self.loc_idx+1][1].split(':')[-1])
        if time == cum_sec:
            self.loc_idx += 1
            self.loc = [self.mob[self.loc_idx][2], self.mob[self.loc_idx][3]]

#############################  Requests  ###########################
class Request():
    def __init__(self, user_id, edge_id, content_id, cach_list):
        # id
        self.content_id = content_id
        # print('content_id: {}'.format(self.content_id))
        self.user_id = user_id
        self.edge_id = edge_id
        self.edge_loc = 0
        self.delay = 0
        # state
        self.state = 5  # 5: not connect
        self.pre_state = 5
        # transmission size
        self.u2e_size = 0
        self.process_size = 0
        self.e2u_size = 0
        # edge state
        self.resource = 0
        self.mig_size = 0
        # predicted pop
        self.pop = p_table()
        # tasktype
        self.tasktype = TaskType(self.content_id, self.pop, cach_list, self.edge_id)
        self.last_offlaoding = 0
        # timer
        self.timer = 0

class TaskType():
    def __init__(self, content_id, pop, cach_list, edge_id):
        # print(cach_list)
        # task request from UE for the Edge Server
        if len(cach_list) == 0:
            self.req_u2e_size = req_u2e_size
            self.process_loading = process_loading
            self.req_e2u_size = req_u2e_size * 0.1
        else:
            if edge_id < RSU_NUM:
                cache_capacity = CACH_CAP_RSU
            else:
                cache_capacity = CACH_CAP_BS
            n = int(cache_capacity / req_u2e_size)
            if n > CONTENT_TYPE_NUM:
                buffer = [] * int(CONTENT_TYPE_NUM * 0.6)
            else:
                buffer = [] * n
            # caching buffer
            temp = []
            for i in range(len(pop)):
                temp.append(pop[i])
            for i in range(len(cach_list)):
                max_index = temp.index(max(temp))
                if cach_list[max_index] != 0:
                    buffer.append(max_index + 1)
                temp[max_index] = -1
            # required data is cached
            # print(content_id)
            if content_id in buffer:
                self.req_u2e_size = 0
                self.process_loading = 0
                self.req_e2u_size = req_u2e_size * 0.1
                # print('cached!')
            # required data is not cached, such that computational task need to be offloaded for execution
            else:
                self.req_u2e_size = req_u2e_size
                self.process_loading = process_loading
                self.req_e2u_size = req_u2e_size * 0.1

    def task_inf(self):
        return "req_u2e_size:" + str(self.req_u2e_size) + "\nprocess_loading:" + str(
            self.process_loading) + "\nreq_e2u_size:" + str(self.req_e2u_size)


#############################Edge Server###################
class EdgeServer():
    def __init__(self, edge_id, loc, e_type):
        self.edge_id = edge_id  # edge server number
        self.e_type = e_type    # bs or rsu
        self.loc = loc
        self.capability = R_BOUND
        self.user_group = []
        self.limit = LIMIT
        self.connection_num = 0

        if self.e_type == 'bs':
            self.limit *= 2
            self.capability *= 2

    def maintain_request(self, R, U):
        for user in U:
            # the number of the connection user
            self.connection_num = 0
            for user_id in self.user_group:
                if U[user_id].req.state != 6:
                    self.connection_num += 1
            # maintain the request
            if user.req.edge_id == self.edge_id and self.capability - R[user.user_id] > 0:
                # maintain the preliminary connection
                if user.req.user_id not in self.user_group and self.connection_num + 1 <= self.limit:
                    # first time : do not belong to any edge(user_group)
                    self.user_group.append(user.user_id)  # add to the user_group
                    user.req.state = 0  # prepare to connect
                    # notify the request
                    user.req.edge_id = self.edge_id
                    user.req.edge_loc = self.loc

                # dispatch the resource
                user.req.resource = R[user.user_id]
                self.capability -= R[user.user_id]

    def release(self, resource):
        self.capability += resource

    # release the all resource
    def releaseAll(self):
        if self.e_type == 'rsu':
            self.capability = R_BOUND
        else:
            self.capability = R_BOUND * 2

# use at the initialization
#############################Policy#######################
class priority_policy():
    def generate_priority(self, U, E, priority):
        for user in U:
            # get a list of the offloading priority
            dist = np.zeros(EDGE_NUM)
            for edge in E:
                dist[edge.edge_id] = geodistance(user.loc[0], user.loc[1], edge.loc[0], edge.loc[1])
            dist_sort = np.sort(dist)
            for index in range(EDGE_NUM):
                priority[user.user_id][index] = np.argwhere(dist == dist_sort[index])[0]
        return priority

    def indicate_edge(self, O, U, priority):
        edge_limit = np.ones((EDGE_NUM)) * LIMIT
        for user in U:
            for index in range(EDGE_NUM):
                if edge_limit[int(priority[user.user_id][index])] - 1 >= 0:
                    edge_limit[int(priority[user.user_id][index])] -= 1
                    O[user.user_id] = priority[user.user_id][index]
                    break
        return O

    def indicate_edge_random(self, O, U):
        for user in U:
            O[user.user_id] = random.randint(0, EDGE_NUM - 1)
        return O

    def indicate_caching(self, C):
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C[i][j] = random.random()
        return C

    def resource_update(self, R, E, U):
        for edge in E:
            # count the number of the connection user
            connect_num = 0
            for user_id in edge.user_group:
                if U[user_id].req.state != 5 and U[user_id].req.state != 6:
                    connect_num += 1
            # dispatch the resource to the connection user
            for user_id in edge.user_group:
                # no need to provide resource to the disconnecting users
                if U[user_id].req.state == 5 or U[user_id].req.state == 6:
                    R[user_id] = 0
                # provide resource to connecting users
                else:
                    R[user_id] = edge.capability  # reserve the resource to those want to migration
        return R

    def bandwidth_update(self, O, B, U, E):
        for user in U:
            ini_edge = int(user.req.edge_id)
            target_edge = int(O[user.req.user_id])
            # no need to migrate
            if ini_edge == target_edge:
                if E[target_edge].e_type == 'rsu':
                    B[user.req.user_id] = random.uniform(0, B_BOUND)
                else:
                    B[user.req.user_id] = random.uniform(0, B_BOUND*2)
        return B


#############################Env###########################
class Env():
    def __init__(self):
        self.time = 0
        self.rsu_num = RSU_NUM  # the number of rsus
        self.bs_num = BS_NUM
        self.edge_num = self.rsu_num+self.bs_num
        self.sdv_num = SDV_NUM  # the number of sdvs
        self.type_num = CONTENT_TYPE_NUM    # the type of content
        # define environment object
        self.reward_all = []
        self.U = []
        self.fin_req_count = 0
        self.prev_count = 0
        self.rewards = 0
        self.R = np.zeros((self.sdv_num))
        self.O = np.zeros((self.sdv_num))
        self.B = np.zeros((self.sdv_num))
        self.P = np.zeros(self.type_num)
        self.table = BandwidthTable(self.edge_num)
        self.priority = np.zeros((self.sdv_num, self.edge_num))
        # self.x_min, self.y_min = get_minimum()
        self.f_type = CONTENT_TYPE_NUM
        self.C = np.zeros((self.edge_num, self.f_type))

        self.e_l = 0
        self.model = 0

        self.done = False

    def get_inf(self):
        # s_dim
        self.reset()
        s = generate_state(self.table, self.U, self.E)
        s_dim = s.size

        # a_dim
        r_dim = len(self.U)
        b_dim = len(self.U)
        o_dim = (self.edge_num) * len(self.U)
        f_dim = self.f_type
        c_dim = f_dim * (self.edge_num)

        # maximum resource
        r_bound = self.E[0].capability

        # maximum bandwidth
        b_bound = self.table[0][1]
        b_bound = b_bound.astype(np.float32)

        # # task size
        # task = TaskType()
        # task_inf = task.task_inf()

        return s_dim, r_dim, b_dim, o_dim, f_dim, c_dim, r_bound, b_bound, LIMIT

    def get_hyperparameters(self):
        return SDV_NUM, EDGE_NUM, req_u2e_size / Mb / 8, process_loading / Mcyc

    def reset(self):
        # reset time
        self.time = 0
        # reward
        self.reward_all = []
        # user
        self.U = []
        self.fin_req_count = 0
        self.prev_count = 0
        moves = pd.read_csv('Movement/selected_sdv.txt', header=None).values.tolist()     # load the movement of selected vehicles
        data_idxs = random.sample(moves, self.sdv_num)
        # print(data_idxs)
        for i in range(self.sdv_num):
            new_user = UE(i, data_idxs[i][0])
            self.U.append(new_user)
        # Resource
        self.R = np.zeros((self.sdv_num))
        # Offlaoding
        self.O = np.zeros((self.sdv_num))
        # bandwidth
        self.B = np.zeros((self.sdv_num))
        # bandwidth table
        self.table = BandwidthTable(self.edge_num)
        # popularity
        self.P = np.zeros(self.type_num)
        # Caching
        self.C = np.zeros((self.edge_num, self.f_type))
        # edge servers
        self.E = []
        e_l = proper_edge_loc(self.rsu_num)    # rsu
        for i in range(self.rsu_num):
            new_e = EdgeServer(i, e_l[i], 'rsu')
            self.E.append(new_e)
        bs_l = proper_bs_loc()                  # bs
        for i in range(self.bs_num):
            new_bs = EdgeServer(i+self.rsu_num, bs_l[i], 'bs')
            self.E.append(new_bs)
        # models
        self.model = priority_policy()

        # initialize the request
        self.priority = self.model.generate_priority(self.U, self.E, self.priority)
        self.O = self.model.indicate_edge(self.O, self.U, self.priority)
        for user in self.U:
            user.generate_request(self.O[user.user_id], [])

        self.done = False

        return generate_state(self.table, self.U, self.E)
    
    def step_forward(self, a, r_dim, b_dim):
        a = np.nan_to_num(a)
        aver_t = 0  # average delay
        # release the bandwidth
        self.table = BandwidthTable(self.edge_num)
        # release the resource
        for edge in self.E:
            edge.releaseAll()

        # update the policy every second
        # resource update
        self.R = a[:r_dim]
        # print(self.R)
        # bandwidth update
        self.B = a[r_dim:r_dim + b_dim]
        # print(self.B)
        # offloading update
        base = r_dim + b_dim
        # offloading decisions
        for user_id in range(self.sdv_num):
            prob_weights = a[base:base + self.edge_num]
            # print(prob_weights)
            # print(sum(prob_weights))
            # print(a[base:base + self.edge_num])
            action = np.random.choice(range(len(prob_weights)),
                                    p=prob_weights.ravel())  # select action w.r.t the actions prob
            base += self.edge_num
            self.O[user_id] = action
        # print(self.O)
        # caching decisions
        for edge_id in range(self.edge_num):
            # print(a[base:base + self.f_type])
            a[base:base + self.f_type]
            self.C[edge_id] = list(map(int, a[base:base + self.f_type]))  # to Integer
            # print(self.C[edge_id])
            base += self.f_type

        # edge update
        for edge in self.E:
            edge.maintain_request(self.R, self.U)

        # request update
        for user in self.U:
            # update the state of the request
            if self.B[user.user_id] != 0:
                user.request_update(self.B[user.user_id])
            else:
                if (user.req.state == 1 and user.req.u2e_size <= 0) or (
                        user.req.state == 2 and user.req.process_size > 0):
                    user.request_update(0)
            # print("bandwidth = ", self.B[user.user_id])
            if user.req.timer >= 1:
                user.generate_request(self.O[user.user_id],
                                    self.C[int(self.O[user.user_id])])  # offload according to the priority
            # it has already finished the request
            if user.req.state == 4:
                # rewards
                self.fin_req_count += 1
                aver_t += user.req.delay
                # print(aver_t)
                user.req.state = 5  # request turn to "disconnect"
                # print("resource:", user.req.resource)
                # self.E[int(user.req.edge_id)].release(user.req.resource)
                self.E[int(user.req.edge_id)].user_group.remove(user.req.user_id)
                user.generate_request(self.O[user.user_id],
                                      self.C[int(self.O[user.user_id])])  # offload according to the priority

        # rewards
        self.rewards = -aver_t / LIMIT

        # update time (second)
        self.time += 1

        # each user starts to move
        for user in self.U:
            user.mobility_update(self.time)

        if self.time == 900:
            self.done = True

        # return s_, r, done
        return generate_state(self.table, self.U, self.E), self.rewards, self.done