from env.my_env import Env
from models.myDDPG import DDPG
import numpy as np
import time
from utils.configs import *

#####################  hyper parameters  ####################
LEARNING_MAX_EPISODE = 1000
CHANGE = True
#############################################################

#####################  exploration  ####################
def exploration (a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a

###############################  training  ####################################
if __name__ == "__main__":
    begin = time.time()
    env = Env()
    ##################################
    s_dim, r_dim, b_dim, o_dim, f_dim, c_dim, r_bound, b_bound, _ = env.get_inf()
    ddpg = DDPG(s_dim, r_dim, b_dim, o_dim, f_dim, c_dim, r_bound, b_bound)
    ##################################
    lr = ddpg.learn_rate
    sdv_num, edge_num, input_size, cpu_freq = env.get_hyperparameters()

    r_var = 1  # control exploration
    b_var = 1
    ep_reward = []
    r_v, b_v = [], []
    var_reward = []
    max_rewards = 0
    episode = 0
    var_counter = 0
    epoch_inf = []
    res = []
    while episode < LEARNING_MAX_EPISODE:
        # initialize
        s = env.reset()
        # print('s', len(s))
        done = False
        ep_reward.append(0)
        n = 0
        delay = []
        # start training
        while True:
            # choose action according to state
            a = ddpg.choose_action(s)  # a = [R B O C]
            # print(a.shape)
            # add randomness(noise) to action selection for exploration
            a = exploration(a, r_dim, b_dim, r_var, b_var)
            # print('a', len(a))
            # print(a)
            s_, r, done = env.step_forward(a, r_dim, b_dim)
            # delay
            if r != 0:
                delay.append(r)
            ddpg.store_transition(s, a, r, s_)
            # learn
            if ddpg.pointer > ddpg.memory_capacity:
                ddpg.learn()
                if CHANGE and r_var >= 0.5:
                    r_var *= .9999
                    b_var *= .9999
            # replace the state
            s = s_
            # in the end of the episode
            if done:
                var_reward.append(ep_reward[episode])
                r_v.append(r_var)
                b_v.append(b_var)
                print('Episode:%3d' % episode, ' | reward %.3f'%np.mean(delay, axis=0),
                      '###  r_var: %.2f ' % r_var,'b_var: %.2f ' % b_var, )
                print('time cost: ' , -1*np.mean(delay, axis=0))
                break

        res.append(-np.mean(delay, axis=0))
        episode += 1

    print('TIME: {:.2f}m'.format((time.time() - begin)/60))