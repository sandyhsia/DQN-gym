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
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
CHECKPOINT_DIR = './checkpoint'
save_request = 1
restore_request = 1

def main():
    start_time = time.time()
    # initialize OpenAI Gym env and dqn agent
    env = Virtual_Env(ENV_NAME, 640, 480)
    agent = DQN(env)
    saver = tf.train.Saver()
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
                            time.sleep(0.01)


                    # Test every 100 episodes
                    if episode % 100 == 0:
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
                                            time.sleep(0.01)
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
                            f.write(str(wall_t))
                    saver.save(agent.session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step =global_t+episode)
                    print "Save done."

if __name__ == '__main__':
        main()