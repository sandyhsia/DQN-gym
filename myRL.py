import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from Virtual_Env import *
from car_DQN import *

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'Exp-0'
EPISODE = 100000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
FC_CHECKPOINT_DIR = './checkpoint/Use_fc'
Conv_CHECKPOINT_DIR = './checkpoint/Use_conv'
LSTM_CHECKPOINT_DIR = './checkpoint/Use_lstm'
USE_FC_ONLY = 0 # state is a 362-dim vector
USE_CONV = 1 # state is a 20x20-dim vector
USE_LSTM = 0 # adding rnn to the DQN
save_request = 1
restore_request = 1
want_test = 0

def main():
    start_time = time.time()
    # initialize OpenAI Gym env and dqn agent
    env = Virtual_Env(ENV_NAME, 100, 100)
    agent = DQN(env)
    saver = tf.train.Saver()
    global_t = 0
    if USE_FC_ONLY == 1:
            CHECKPOINT_DIR = FC_CHECKPOINT_DIR
    elif USE_CONV == 1:
            CHECKPOINT_DIR = Conv_CHECKPOINT_DIR
    elif USE_LSTM == 1:
            CHECKPOINT_DIR = LSTM_CHECKPOINT_DIR

    if restore_request == 1:
            checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(agent.session, checkpoint.model_checkpoint_path)
                    tokens = checkpoint.model_checkpoint_path.split("-")
                    # set global step
                    global_t = int(tokens[1])
                    print(">>> global step set: ", global_t)
                    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            else:
                    print("Could not find old checkpoint")

    try:
            for episode in range(EPISODE):
                    # initialize task
                    print "episode", episode
                    state = env.reset()
                    time.sleep(0.5)
                    # Train
                    for step in range(STEP):
                            action = agent.egreedy_action(state) # e-greedy action for train
                            # print action
                            next_state,reward,done = env.step(env.car_center, env.angle, action)
                            # Define reward for agent
                            agent.perceive(state,action,reward,next_state,done)
                            state = next_state
                            if done == True:
                                    break
                            # time.sleep(0.01)


                    # Test every 100 episodes
                    if episode % 100 == 0 and want_test:
                            print "---Test---"
                            total_reward = 0
                            for i in xrange(TEST):
                                    state = env.reset()
                                    time.sleep(0.5)
                                    for j in xrange(STEP):
                                            action = agent.action(state) # direct action for test
                                            state,reward,done= env.step(env.car_center, env.angle, action)
                                            total_reward += reward
                                            if done == True:
                                                    break
                                            # time.sleep(0.01)
                            ave_reward = total_reward/TEST
                            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
                            if ave_reward >= 1000:
                                    break
            
            print "----Average reward reaches expectation. Exit training.----"
            if save_request == 1:
                    print "Now saving... wait."
                    if not os.path.exists(CHECKPOINT_DIR):
                            os.mkdir(CHECKPOINT_DIR) 
                    # write wall time
                    wall_t = time.time() - start_time
                    wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t+episode)
                    with open(wall_t_fname, 'w') as f:
                            f.write(time.ctime())
                            f.write(str(wall_t))
                    saver.save(agent.session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step =global_t+episode)
                    print "Save done."


    except KeyboardInterrupt:
            print "----You press Ctrl+C.----"
            if save_request == 1:
                    print "Now saving... wait."
                    if not os.path.exists(CHECKPOINT_DIR):
                            os.mkdir(CHECKPOINT_DIR)  
                    # write wall time
                    wall_t = time.time() - start_time
                    wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t+episode)
                    with open(wall_t_fname, 'w') as f:
                            f.write(time.ctime())
                            f.write(str(wall_t))
                    saver.save(agent.session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step =global_t+episode)
                    print "Save done."

if __name__ == '__main__':
        main()