import tensorflow as tf
import numpy as np
import random
from collections import deque
import time

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.05 # starting value of epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
REPLAY_SIZE = 100000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
DEBUG_MODE = 0 # print some info and slow down.
USE_FC_ONLY = 1 # state is a 362-dim vector
USE_CONV = 0 # state is a 20x20-dim vector
USE_LSTM = 0 # adding rnn to the DQN
pack_size = 20 # pack one-dim vector into size*size*1 "img" so as to CONV

class DQN():
        #DQN Agent
        def __init__(self, env):
                
                # ''env'' for car agent should be virtual sensor playground
                # env.observation should be 360 degree vec
                # env.action_space should be discrete action like 'wasd' x '0123' + 'q3' = 17

                # init experience replay
                self.replay_buffer = deque()
                # init some parameters
                self.time_step = 0
                self.epsilon = INITIAL_EPSILON
                self.state_dim = 362
                self.action_dim = 3

                self.create_Q_network()
                self.create_training_method()

                self.session = tf.InteractiveSession()
                self.session.run(tf.initialize_all_variables())
                self.time_t = 0
                self.train_time = 1
                self.loss = 0


        def create_Q_network(self):
                
                if USE_FC_ONLY == 1:
                        # network weights
                        W1 = self.weight_variable([self.state_dim, 64])
                        b1 = self.bias_variable([64])
                        W2 = self.weight_variable([64, 20])
                        b2 = self.bias_variable([20])

                        W3 =  self.weight_variable([20, self.action_dim])
                        b3 = self.bias_variable([self.action_dim])

                        if DEBUG_MODE:
                                print "All weights and bias:"
                                print "W1: ", W1
                                print "b1: ", b1
                                print "W2: ", W2
                                print "b2: ", b2
                                print "W3:", W3
                                print "b3: ", b3
                                time.sleep(0.5)

                        #input layer
                        self.state_input = tf.placeholder("float", [None, self.state_dim])
                        # hidden layers
                        h1_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
                        h2_layer = tf.nn.relu(tf.matmul(h1_layer, W2) + b2)
                        # Q Value layer
                        self.Q_value = tf.matmul(h2_layer, W3)+b3

                elif USE_CONV == 1:
                        # network weights
                        W_conv1, b_conv1 = self._conv_variable([3, 3, 1, 10])  # stride=2
                        W_conv2, b_conv2 = self._conv_variable([3, 3, 10, 100]) # stride=2

                        W_fc1, b_fc1 = self._fc_variable([1600, 32])

                        # weight for policy output layer
                        W_fc2, b_fc2 = self._fc_variable([32, self.action_dim])

                        # weight for value output layer
                        # self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

                        # state (input)
                        self.state_input = tf.placeholder("float", [None, pack_size, pack_size, 1])
    
                        h_conv1 = tf.nn.relu(self._conv2d(self.state_input,  W_conv1, 2) + b_conv1)
                        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)

                        h_conv2_flat = tf.reshape(h_conv2, [-1, 1600])
                        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

                        # policy (output)
                        self.Q_value = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
                        
                        # value (output)
                        # v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
                        # self.v = tf.reshape( v_, [-1] )

                        if DEBUG_MODE:
                                print "All weights and bias:"
                                print "W_conv1: ", W_conv1
                                print "b_conv1: ", b_conv1
                                time.sleep(0.5)

                elif USE_LSTM == 1:
                        # network weights
                        W_conv1, b_conv1 = self._conv_variable([3, 3, 1, 10])  # stride=2
                        W_conv2, b_conv2 = self._conv_variable([3, 3, 10, 100]) # stride=2

                        W_fc1, b_fc1 = self._fc_variable([1600, 256])

                        # weight for policy output layer
                        W_fc2, b_fc2 = self._fc_variable([256, self.action_dim])

                        # weight for value output layer
                        # self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

                        # state (input)
                        self.state_input = tf.placeholder("float", [None, pack_size, pack_size, 1])
    
                        h_conv1 = tf.nn.relu(self._conv2d(self.state_input,  W_conv1, 2) + b_conv1)
                        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)

                        h_conv2_flat = tf.reshape(h_conv2, [-1, 1600])
                        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
                        h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])

                        # lstm
                        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)

                        # place holder for LSTM unrolling time step size.
                        self.step_size = tf.placeholder("float", [1])

                        self.initial_lstm_state0 = tf.placeholder("float", [1, 256])
                        self.initial_lstm_state1 = tf.placeholder("float", [1, 256])
                        self.initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_lstm_state0,
                                                              self.initial_lstm_state1)
      
                        # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
                        # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
                        # Unrolling step size is applied via self.step_size placeholder.
                        # When forward propagating, step_size is 1.
                        # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
                        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = self.step_size,
                                                        time_major = False)
                                                        #scope = scope)

                        # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
      
                        lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

                        # policy (output)
                        self.Q_value = tf.nn.softmax(tf.matmul(lstm_outputs, W_fc2) + b_fc2)



        def create_training_method(self):
                self.action_input = tf.placeholder("float", [None, self.action_dim]) # one hot presentation
                self.y_input = tf.placeholder("float", [None])
                Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices = 1)
                self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
                self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

        def perceive(self, state, action, reward, next_state, done):
                one_hot_action = np.zeros(self.action_dim)
                one_hot_action[action] = 1
                self.time_t += 1
                
                if USE_FC_ONLY == 1:
                        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
                
                elif USE_CONV == 1:
                        packed_state = self.pack_up(state, pack_size)
                        packed_next_state = self.pack_up(next_state, pack_size)
                        self.replay_buffer.append((packed_state, one_hot_action, reward, packed_next_state, done))

                if len(self.replay_buffer) > REPLAY_SIZE:
                        self.replay_buffer.popleft()

                if len(self.replay_buffer) > BATCH_SIZE*2 and self.time_t%self.train_time == 0:
                        self.train_Q_network()
                        self.time_t = 0

        def train_Q_network(self):

                self.time_step += 1
                # Step 1: obtain random minibatch from replay memory
                minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
                state_batch = [data[0] for data in minibatch] # refer to append order
                action_batch = [data[1] for data in minibatch]
                reward_batch = [data[2] for data in minibatch]
                next_state_batch = [data[3] for data in minibatch]

                if DEBUG_MODE:
                        print "Training. Please wait." 
                        print "Fetch minibatch[0] to check: ---------------"
                        print "state_batch: ", state_batch[0][17:19]
                        # print "action_batch: ", action_batch[0]
                        print "reward_batch: ", reward_batch[0]
                        # print "next_state_batch: ", next_state_batch[0]
                        print "done? : ", minibatch[0][4]
                        time.sleep(0.5)

                # Step 2: calculate y
                y_batch = []
                Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
                for i in range(0, BATCH_SIZE):
                        done = minibatch[i][4]
                        if done:
                                y_batch.append(reward_batch[i])
                        else:
                                y_batch.append(reward_batch[i]+GAMMA*np.max(Q_value_batch[i]))
                
                self.optimizer.run(feed_dict={
                            self.y_input: y_batch,
                            self.action_input: action_batch,
                            self.state_input: state_batch
                    })

                loss = 0
                for i in range(0, BATCH_SIZE):
                        loss += (Q_value_batch[i][np.argmax(Q_value_batch[i])] - y_batch[i])**2
                self.loss += loss/BATCH_SIZE

                if DEBUG_MODE:
                        print "calculate y..." 
                        print "Q_value_batch: ", Q_value_batch[0]
                        print "y value: ", y_batch[0]
                        print "loss: ", loss/BATCH_SIZE
                        time.sleep(0.5)

        def egreedy_action(self, state):
                if USE_FC_ONLY == 1:
                        Q_value = self.Q_value.eval(feed_dict={
                            self.state_input: [state]
                            })[0] #?[0]
                elif USE_CONV == 1:
                        Q_value = self.Q_value.eval(feed_dict={
                            self.state_input: [self.pack_up(state, pack_size)]
                            })[0]

                elif USE_LSTM == 1:
                        Q_value = self.Q_value.eval(feed_dict={
                            self.state_input: [self.pack_up(state, pack_size)]
                            })[0]

                if DEBUG_MODE:
                        print "egreedy_action -- Q_value:", Q_value
                        time.sleep(0.5)


                if random.random() <= self.epsilon:
                        self.epsilon -= (self.epsilon - FINAL_EPSILON)/10000
                        ## Note!!!!
                        ## Initially, self.epsilon -= (INITIAL_EPSILON - FIANAL_EPSILON)/10000
                        ## But actually, result is after 10000/300 ~= 33 episode, exploration rate is 0.01
                        return random.randint(0, self.action_dim -1)
                
                else:
                        self.epsilon -= (self.epsilon - FINAL_EPSILON)/10000
                        return np.argmax(Q_value)

        def action(self, state):
                if USE_FC_ONLY == 1:
                        Q_value = self.Q_value.eval(feed_dict = {
                            self.state_input:[state]
                            })[0]
                elif USE_CONV == 1:
                        Q_value = self.Q_value.eval(feed_dict={
                            self.state_input: [self.pack_up(state, pack_size)]
                            })[0]

                elif USE_LSTM == 1:
                        Q_value = self.Q_value.eval(feed_dict={
                            self.state_input: [self.pack_up(state, pack_size)]
                            })[0]

                if DEBUG_MODE:
                        print "action -- Q_value:", Q_value
                        time.sleep(0.5)

                action = np.argmax(Q_value)
                return action

        def normalize(self, input):
                sum_up = 0

                if DEBUG_MODE:
                        print "abefore normalize, input:", input
                        time.sleep(0.5)

                for i in range(0, 360):
                        input[i] = input[i]/50

                input[360] = input[360]/10000
                input[361] = input[361]/180

                if DEBUG_MODE:
                        print "after normalize, input:", input
                        time.sleep(0.5)
                
                return input


        def weight_variable(self, shape):
                # initial = tf.truncated_normal(shape)
                return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

        def bias_variable(self, shape):
                initial = tf.constant(0.01, shape = shape)
                return tf.Variable(initial)

        def _fc_variable(self, weight_shape):
                input_channels  = weight_shape[0]
                output_channels = weight_shape[1]
                d = 1.0 / np.sqrt(input_channels)
                bias_shape = [output_channels]
                weight = tf.Variable((tf.truncated_normal(weight_shape, stddev = 0.1)))
                bias   = tf.Variable((tf.truncated_normal(bias_shape, stddev = 0.1)))
                return weight, bias

        def _conv_variable(self, weight_shape):
                w = weight_shape[0]
                h = weight_shape[1]
                input_channels  = weight_shape[2]
                output_channels = weight_shape[3]
                d = 1.0 / np.sqrt(input_channels * w * h)
                bias_shape = [output_channels]
                weight = tf.Variable((tf.truncated_normal(weight_shape, stddev = 0.1)))
                bias   = tf.Variable((tf.truncated_normal(bias_shape, stddev = 0.1)))
                return weight, bias

        def _conv2d(self, x, W, stride):
                return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

        def _maxpool(self, x, ksize, pool_strides):
                maxpooling_layer = tf.nn.max_pool(x,
                                ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                                strides=[1, pool_strides[0], pool_strides[1], 1],
                                padding='SAME')
                return maxpooling_layer

        def pack_up(self, state, size):
                pack_state = np.zeros(((size, size, 1)))
                # print state[0:10]

                for i in range(len(state)):
                        pack_state[i/size][i%size] = [state[i]]

                # print pack_state[0:2]
                return pack_state

        def reset_loss(self):
                self.loss = 0
                return 1

        def get_loss(self):
                return self.loss


