import numpy as np
import random
import gym
import sys
import math


EPISODES = 5000
LEARNING_RATE = 0.2
DISCOUNT = 1
EPSILON = 1
EPSILON_DECAY = 0.997
import pandas as pd


env = gym.make('CartPole-v1')


cart_position_bins = pd.cut([env.observation_space.low[0], env.observation_space.high[0]], bins=5, retbins=True)[1]
cart_velocity_bins = pd.cut([env.observation_space.low[1], env.observation_space.high[1]], bins=5, retbins=True)[1]
pole_angle_bins = pd.cut([env.observation_space.low[2], env.observation_space.high[2]], bins=10, retbins=True)[1]
pole_angular_velocity_bins = pd.cut([env.observation_space.low[3], env.observation_space.high[3]], bins=10, retbins=True)[1]

def discretize(state):
    return (pd.cut([state[0]], cart_position_bins, labels=[0,1,2,3,4])[0], 
            pd.cut([state[1]], cart_velocity_bins, labels=[0,1,2,3,4])[0], 
            pd.cut([state[2]], pole_angle_bins, labels=[0,1,2,3,4,5,6,7,8,9])[0], 
            pd.cut([state[3]], pole_angular_velocity_bins, labels=[0,1,2,3,4,5,6,7,8,9])[0])


if __name__ == "__main__":
    q = np.zeros((10,10,10,10,2), dtype=float)
    qq = np.zeros((10,10,10,10,2,2), dtype=float)

    print(env.observation_space.low)

    epsilon = EPSILON
    for episode in range(EPISODES):
        state = env.reset()
        state = discretize(state)
        done = False
        episode_rewards = 0
        
        episode_memory = [] # stores the (state, action, reward) tuple

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])

            #PERFORM ACTION & COLLECT NEW STATE, REWARD, and DONE
            next_state, reward, done, info = env.step(action)
            
            #ADJUST AND ROUND NEW STATE
            next_state = discretize(next_state)
            
            #UPDATE
            episode_memory.append((state, action, reward))
            
            #ADD REWARD TO EPISODE_REWARDS
            episode_rewards += reward
            
            #set state to next_state
            state = next_state
        
        
        # calculate discounted_reward based by iterating backward through `episode_memory_sar`
        discounted_reward = 0
        for state, action, reward in episode_memory[::-1]:
            discounted_reward = reward + (DISCOUNT * discounted_reward)      
            qq[state][action][0] += 1
            qq[state][action][1] += discounted_reward
            q[state][action] = qq[state][action][1]/qq[state][action][0]
            #q[state][action] = q[state][action] + (LEARNING_RATE * (discounted_reward - q[state][action]))
                    
        #Decay Epsilon
        epsilon = max(epsilon * EPSILON_DECAY, 0.1)
        
        #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS
        print("EPISODE: {}    REWARD: {}    EPSILON: {}".format(episode, episode_rewards, epsilon))
