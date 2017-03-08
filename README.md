# DQN-gym
Experiments on DQN

### Environment request:
> Tensorflow 0.10.0 

> (Other version might need you change some function usages)

## Playing with your AI

### 1. To train:

> python ./myRL.py

### 2. To display:

> python ./myRL-display.py

### 3. Hyper-params in **myRL.py** or **myRL-display.py**:

> CHECKPOINT_DIR = './checkpoint

where you can save to, or load from check point.


> save_request = 1

means you're sure your environment is ok, thus you want to save your model after a period of training.


> restore_request = 1

means you want to train from "this" check point file (clarified in './checkpoint/checkpoint' file)
Usaully, the file contain the lastest checkpoint file name.
However, in another way around, if you change the the file name here, then, it would start from the place you want.

### 4. Where to change:

> 1. (Strongly recommend...) Every hyper-params in **car-DQN.py** and Q-network structure here.

> 2. reward_method() in **Virtual-Env.py**

> 3. In **car-DQN.py**, I am still not sure how to manipulate the replay_buffer(). And I am trying to add so-called memory and attention to agent.

Idea from here: https://zhuanlan.zhihu.com/p/21320865?refer=intelligentunit



