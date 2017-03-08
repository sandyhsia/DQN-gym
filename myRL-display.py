import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from Virtual_Env import *
from car_DQN import *

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'MyRL-Display'
TEST_EPISODE = 5 # Episode limitation
STEP = 3000 # Step limitation in an episode
CHECKPOINT_DIR = './checkpoint'

def main():
    start_time = time.time()
    # initialize OpenAI Gym env and dqn agent
    env = Virtual_Env(ENV_NAME, 640, 480)
    agent = DQN(env)

    saver = tf.train.Saver()
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
            total_reward = 0
            for episode in range(TEST_EPISODE):
                    print "episode", episode
                    state = env.reset()
                    time.sleep(0.5)
                    for j in xrange(STEP):
                            action = agent.action(state) # direct action for test
                            state,reward,done= env.step(env.car_center, env.angle, action)
                            total_reward += reward
                            if done == True:
                                    print "Done in step", j
                                    print "Average reward in previous ", (episode+1), "episode: ", total_reward/(episode+1)
                                    break
                            time.sleep(0.01)

            ave_reward = total_reward/TEST_EPISODE
            
            print "----Exit display.----"
            print "No saving..."


    except:
            print "----You press Ctrl+C.----"
            print "----Exit display.----"
            print "No saving..."

if __name__ == '__main__':
        main()