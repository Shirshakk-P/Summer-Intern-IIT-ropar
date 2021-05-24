#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np

from gym.envs.registration import register
register(
    id='Deterministic-4x4-FrozenLake-v0', #new environment
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv', 
    kwargs={'map_name': '4x4', 'is_slippery': False} # argument passed to the env
)
env = gym.make('Deterministic-4x4-FrozenLake-v0')
my_desk = [
    "GFFFF",
    "FFFFF",
    "FFFFG",
    "FFFFF",
    "FGFFG"
]
 


import gym

class CustomizedFrozenLake(gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
    def __init__(self, **kwargs):
        super(CustomizedFrozenLake, self).__init__(**kwargs)

        for state in range(self.nS): # for all states
            for action in range(self.nA): # for all actions
                my_transitions = []
                for (prob, next_state, _, is_terminal) in self.P[state][action]:
                    row = next_state // self.ncol
                    col = next_state - row * self.ncol
                    tile_type = self.desc[row, col]
                    if tile_type == b'F':
                        reward = -1
                    elif tile_type == b'G':
                        reward = 10
                    #else:
                        #reward = 0

                    my_transitions.append((prob, next_state, reward, is_terminal))
                self.P[state][action] = my_transitions

from gym.envs.registration import register

register(
    id='Stochastic-5x5-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'desc': my_desk, 'is_slippery': False})
env = gym.make('Stochastic-5x5-FrozenLake-v0')
env.render()


# In[ ]:


env.reset()
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))


# In[ ]:


"""ACTIONS DEFINED VIA:
    0 = SOUTH
    1 = NORTH
    2 = EAST
    3 = WEST
"""

state=env.s 
if state in range (0,14):
  print("State:", state)
elif state in range (14,20):
  print("State:", state+1)  
elif state in range (20,22):
  print("State:", state+2)

env.render()


# In[ ]:


env.P[state][1]


# In[ ]:


m=int(input("Enter State numnber for start:"))

if m in range (0,14):
  env.s = m 
elif m in range (14,20):
  env.s = m+1
elif m in range (20,22):
  env.s = m+2
 # set environment to illustration's state
env.render()

print("....................Learning Starts........................")

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -1:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame' : env.render(mode='human'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))
print("\n")
print(frames)


# In[ ]:


from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print("frame: ",frame)
        print(f"Timestep: {i + 1}")
        #print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)


# In[ ]:


"""QLearning"""
import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(q_table)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Training the Agent\n\nimport random\nfrom IPython.display import clear_output\n\n# Hyperparameters\nalpha = 0.1\ngamma = 0.6\nepsilon = 0.1\n\n# For plotting metrics\nall_epochs = []\nall_penalties = []\n\nfor i in range(1, 10001):\n    state = env.reset()\n\n    epochs, penalties, reward, = 0, 0, 0\n    done = False\n    \n    while not done:\n        if random.uniform(0, 1) < epsilon:\n            action = env.action_space.sample() # Explore action space\n        else:\n            action = np.argmax(q_table[state]) # Exploit learned values\n\n        next_state, reward, done, info = env.step(action) \n        \n        old_value = q_table[state, action]\n        next_max = np.max(q_table[next_state])\n        \n        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)\n        q_table[state, action] = new_value\n\n        if reward == -1:\n            penalties += 1\n\n        state = next_state\n        epochs += 1\n        \n    if i % 100 == 0:\n        clear_output(wait=True)\n        print(f"Episode: {i}")\n\nprint("Training finished.\\n")')


# In[ ]:


m=int(input(print("Enter state value:")))


for m in range (0,14):
  print(q_table[m])
for m in range (14,20):
  print(q_table[(m+1)])
for m in range (20,22):
  print(q_table[(m+2)])


# In[ ]:


q_table[20] #Checking Qvalue for any random state


# In[ ]:


"""Evaluating Agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 520148

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -1:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


# In[ ]:




