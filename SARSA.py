#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np

from gym.envs.registration import register
register(
    id='Deterministic-4x4-FrozenLake-v0', # name given to this new environment
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv', # env entry point
    kwargs={'map_name': '4x4', 'is_slippery': False} # argument passed to the env
)
"""We specify the start state at 1..
This can be reconfiguired as per our requirements """
env = gym.make('Deterministic-4x4-FrozenLake-v0') # load the environment
my_desk = [
    "GSFFF",
    "FFFFF",
    "FFFFG",
    "FFFFF",
    "FGFFG"
]


# In[2]:



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
                    else: 
                        reward = 0
                        

                    my_transitions.append((prob, next_state, reward, is_terminal))
                self.P[state][action] = my_transitions

from gym.envs.registration import register

register(
    id='Stochastic-5x5-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'desc': my_desk, 'is_slippery': False})
env = gym.make('Stochastic-5x5-FrozenLake-v0')
env.render()
print(env.action_space.n)
print(env.observation_space.n)


# In[ ]:


#Parameters
epsilon = 0.9
total_episodes = 5500
max_steps = 100
alpha = 0.70
gamma = 0.75



# In[ ]:


#Initializing the Q-matrix 
Q = np.zeros((env.observation_space.n, env.action_space.n)) 
#print(Q)

#Function to choose the next action 
def choose_action(state): 
	action=0
	if np.random.uniform(0, 1) < epsilon: 
		action = env.action_space.sample() 
	else: 
		action = np.argmax(Q[state, :]) 
	return action 

#Function to learn the Q-value 
def update(state, state2, reward, action, action2): 
	predict = Q[state, action] 
	target = reward + gamma * Q[state2, action2] 
	Q[state, action] = Q[state, action] + alpha * (target - predict) 
  
#print(Q)


# In[16]:


#Initializing the reward 
reward=0

# Starting the SARSA learning 
for episode in range(total_episodes): 
	t = 0
	state1 = env.reset() 
	action1 = choose_action(state1) 

	while t < max_steps: 
		#Visualizing the training 
		env.render() 
		
		#Getting the next state 
		state2, reward, done, info = env.step(action1) 

		#Choosing the next action 
		action2 = choose_action(state2) 
		
		#Learning the Q-value 
		update(state1, state2, reward, action1, action2) 

		state1 = state2 
		action1 = action2 
		
		#Updating the respective vaLues 
		t += 1
		reward += -1
		
		#If at the end of learning process 
		if done: 
			break 


# In[10]:


#Evaluating the performance 
print ("Performace : ", reward/total_episodes) 

#Visualizing the Q-matrix 
print(Q) 
"A positive performance is highly acceptable, given that with every step a penalty of -1 is incured"

