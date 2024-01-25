import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# fixed random seed for reproduce
np.random.seed(42)
tf.set_random_seed(42)

#####################  hyper parameters  ####################
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.1      # soft replacement
BATCH_SIZE = 64
###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, s_dim, r_dim, b_dim, o_dim, f_dim, c_dim, r_bound, b_bound):
        self.learn_rate = LR_A
        self.memory_capacity = 3000
        self.s_dim = s_dim  # dimension of state space
        self.a_dim = r_dim + b_dim + o_dim + c_dim  # dimension of action space
        self.r_dim = r_dim  # computation resources
        self.b_dim = b_dim  # bandwidth resources
        self.o_dim = o_dim  # offloading decisions
        self.c_dim = c_dim  # caching decisions
        self.f_dim = f_dim  # content type
        # self.a_bound
        self.r_bound = r_bound  # upper bound of computation resources
        self.b_bound = b_bound  # upper bound of bandwidth resources
        # S, S_, R
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        # memory
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + self.a_dim + 1), dtype=np.float32)  # s_dim + a_dim + r + s_dim
        self.pointer = 0
        # session
        self.sess = tf.Session(config=config)

        # define the input and output
        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )

        # replaced target parameters with the trainning  parameters for every step
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))
        # update the weight for every step
        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        # Actor learn()
        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        # Critic learn()
        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())


    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            n_l = 25
            net = tf.layers.dense(s, n_l, activation=tf.nn.relu, name='l1', trainable=trainable)
            # resource ( 0 - r_bound)
            layer_r0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='r_0', trainable=trainable)
            layer_r1 = tf.layers.dense(layer_r0, n_l, activation=tf.nn.relu, name='r_1', trainable=trainable)
            layer_r2 = tf.layers.dense(layer_r1, n_l, activation=tf.nn.relu, name='r_2', trainable=trainable)
            layer_r3 = tf.layers.dense(layer_r2, n_l, activation=tf.nn.relu, name='r_3', trainable=trainable)
            layer_r4 = tf.layers.dense(layer_r3, self.r_dim, activation=tf.nn.relu, name='r_4', trainable=trainable)

            # bandwidth ( 0 - b_bound)
            layer_b0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='b_0', trainable=trainable)
            layer_b1 = tf.layers.dense(layer_b0, n_l, activation=tf.nn.relu, name='b_1', trainable=trainable)
            layer_b2 = tf.layers.dense(layer_b1, n_l, activation=tf.nn.relu, name='b_2', trainable=trainable)
            layer_b3 = tf.layers.dense(layer_b2, n_l, activation=tf.nn.relu, name='b_3', trainable=trainable)
            layer_b4 = tf.layers.dense(layer_b3, self.b_dim, activation=tf.nn.relu, name='b_4', trainable=trainable)

            # offloading (probability: 0 - 1)
            # layer
            layer = [["layer_o"+str(user_id)+str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # name
            name = [["layer_o"+str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # user
            user = ["user"+str(user_id) for user_id in range(self.r_dim)]
            # softmax
            softmax = ["softmax_o"+str(user_id) for user_id in range(self.r_dim)]
            for user_id in range(self.r_dim):
                layer[user_id][0] = tf.layers.dense(net, n_l, activation=tf.nn.relu, name=name[user_id][0], trainable=trainable)
                layer[user_id][1] = tf.layers.dense(layer[user_id][0], n_l, activation=tf.nn.relu, name=name[user_id][1], trainable=trainable)
                layer[user_id][2] = tf.layers.dense(layer[user_id][1], n_l, activation=tf.nn.relu, name=name[user_id][2], trainable=trainable)
                layer[user_id][3] = tf.layers.dense(layer[user_id][2], (self.o_dim/self.r_dim), activation=tf.nn.relu, name=name[user_id][3], trainable=trainable)
                user[user_id] = tf.nn.softmax(layer[user_id][3], name=softmax[user_id])

            # caching (probability: 0 - 1)
            # layer
            layer = [["layer_c" + str(content_id) + str(layer) for layer in range(4)] for content_id in range(self.f_dim)]
            # name
            name = [["layer_c" + str(content_id) + str(layer) for layer in range(4)] for content_id in range(self.f_dim)]
            for content_id in range(self.f_dim):
                layer[content_id][0] = tf.layers.dense(net, n_l, activation=tf.nn.relu, name=name[content_id][0],trainable=trainable)
                layer[content_id][1] = tf.layers.dense(layer[content_id][0], n_l, activation=tf.nn.relu,name=name[content_id][1], trainable=trainable)
                layer[content_id][2] = tf.layers.dense(layer[content_id][1], n_l, activation=tf.nn.relu,name=name[content_id][2], trainable=trainable)
                layer[content_id][3] = tf.layers.dense(layer[content_id][2], (self.c_dim / self.f_dim),activation=tf.nn.sigmoid, name=name[content_id][3],trainable=trainable)

            # concate
            a = tf.concat([layer_r4, layer_b4], 1)
            for user_id in range(self.r_dim):
                a = tf.concat([a, user[user_id]], 1)
            for content_id in range(self.f_dim):
                a = tf.concat([a, layer[content_id][3]], 1)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # Q value (0 - inf)
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(net_1, n_l, activation=tf.nn.relu, trainable=trainable)
            net_3 = tf.layers.dense(net_2, n_l, activation=tf.nn.relu, trainable=trainable)
            net_4 = tf.layers.dense(net_3, n_l, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net_4, 1, activation=tf.nn.relu, trainable=trainable)  # Q(s,a)
