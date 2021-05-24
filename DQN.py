#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gym
import random

from keras.layers import Dense
from keras.models import Sequential
from gym.envs.registration import register, spec
from collections import deque
from pandas import DataFrame, Series


# In[ ]:


EPISODES = 2048
EPSILON = 0.95
EPSILON_DECAY = 0.95 
EPSILON_MIN = 0.2 
LEARNING_RATE = 0.01 
GAMMA = 0.9 
BATCH_SIZE = 32 #can be customized to 64, better fit with 32 obtained

ACTION_LEFT = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DEFAULT = None
ACTION_TEXT = {
    ACTION_LEFT: 'left',
    ACTION_DOWN: 'down',
    ACTION_RIGHT: 'right',
    ACTION_UP: 'up'
}
"This code has to be rerun in another session as gym library does not allow registering any custom environment twice"
from gym.envs.registration import register
register(
    id='Deterministic-4x4-FrozenLake-v0', # name given to this new environment
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv', # env entry point
    kwargs={'map_name': '4x4', 'is_slippery': False} # argument passed to the env
)
env = gym.make('Deterministic-4x4-FrozenLake-v0') # load the environment
my_desk = [
    "GSFFF",
    "FFFFF",
    "FFFFG",
    "FFFFF",
    "FGFFG"
]
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


class DQNAgent():
    def __init__(self):
        self.env = self._build_env()
        self.nb_status = self.env.observation_space.n
        self.nb_action = self.env.action_space.n
        self.memory = deque(maxlen=2048)
        self.model = self._build_model()

    def _build_env(self): #Customized env setup
        frozen_lake = 'Stochastic-5x5-FrozenLake-v0'
        try:
            spec(frozen_lake)
        except:
            register(id='Stochastic-5x5-FrozenLake-v0',
                     entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
                     kwargs={'desc': my_desc, 'is_slippery': False})
        return gym.make(frozen_lake)

    def episode(self):
        status = self.env.reset()

        while True:
            action = self._choose_action(status)
            next_status, reward, done, info = self.env.step(action)
            self.memory.append((status, action, reward, next_status, done))
            status = next_status

            if done:
                break

    def _choose_action(self, status, choose_best = False, return_probs = False):
        global EPSILON

        if_explore = False
        if choose_best:
            if_explore = False
        else:
            if_explore = np.random.uniform() < EPSILON

        action = ACTION_DEFAULT
        if if_explore:
            # exploration
            action = np.random.choice(self.nb_action)
        else:
            # exploitation
            reward_pred = self.model.predict(self._one_hot_status(status))[0]
            action = np.argmax(reward_pred)

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        return action if not return_probs else (action, reward_pred)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batches = random.sample(self.memory, BATCH_SIZE)
        X = []
        y = []
        for status, action, reward, next_status, done in batches:
            actual_reward = reward

            if not done:
                next_reward_pred = self.model.predict( self._one_hot_status(next_status))
                actual_reward += GAMMA * np.max(next_reward_pred[0])

            one_hot_status = self._one_hot_status(status)
            reward_pred = self.model.predict(one_hot_status)
            reward_pred[0][action] = actual_reward

            X.append(one_hot_status[0])
            y.append(reward_pred[0])

        self.model.train_on_batch(DataFrame(X), DataFrame(y))
        # self.model.fit(X, y, epochs=1, verbose=0)

    def demo(self):
        print("\n------------- DEMO ----------------")
        decisions = []
        rewards = []
        for status in range(self.nb_status):
            best_action, reward = self._choose_action(status, choose_best=True, return_probs=True)
            decisions.append(best_action)
            rewards.append(reward)

        for i in range(self.nb_status):
            text = ''
            if i==1:
                text = 'START'
            elif i in (0,14,21,24):
                text = 'GOAL'
            else:
                text = ACTION_TEXT[decisions[i]]

            print("{0:^7}".format(text), end='')

            if (i + 1) % 5 == 0:
                print('\n')

        print('LEFT\t\tDOWN\t\tRIGHT\t\tUP')
        for r in rewards:
            print([i for i in r])

    def _one_hot_status(self, status):
        one_hot_status = np.zeros(self.nb_status)
        one_hot_status[status] = 1
        one_hot_status = np.expand_dims(one_hot_status, axis=0)
        return one_hot_status

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.nb_status, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.nb_action, activation='linear'))

        model.compile(loss='mse', optimizer='adadelta')
        model.summary()

        return model

def main():
    agent = DQNAgent()

    for i in range(EPISODES):
        agent.episode()
        agent.replay()

        if (i+1) % 512 == 0:
            agent.demo()
            

if __name__ == '__main__':
    main()
    print('\nDone')

