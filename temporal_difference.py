import numpy as np
import random
import gym
import sys
import pickle
import pandas as pd
from os import path


EPISODES = 5000
LEARNING_RATE = 0.2
DISCOUNT = .95
EPSILON = 1
EPSILON_DECAY = 0.997


env = gym.make('CartPole-v1')


cart_position_bins = pd.cut([env.observation_space.low[0], env.observation_space.high[0]], bins=5, retbins=True)[1]
cart_velocity_bins = pd.cut([env.observation_space.low[1], env.observation_space.high[1]], bins=5, retbins=True)[1]
pole_angle_bins = pd.cut([env.observation_space.low[2], env.observation_space.high[2]], bins=10, retbins=True)[1]
pole_angular_velocity_bins = pd.cut([env.observation_space.low[3], env.observation_space.high[3]], bins=10, retbins=True)[1]


def bin(val, bins):
    for i in range(len(bins)):
        if val >= bins[i] and val < bins[i+1]:
            return i

def discretize(state):
    return (bin(state[0], cart_position_bins), bin(state[1], cart_velocity_bins), bin(state[2], pole_angle_bins), bin(state[3], pole_angular_velocity_bins))


if __name__ == "__main__":
    q = np.zeros((5,5,10,10,2), dtype=float)
    total_episode_rewards = []
    epsilon = EPSILON
    for episode in range(EPISODES):
        state = env.reset()
        state = discretize(state)
        done = False
        episode_rewards = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])
            #PERFORM ACTION & COLLECT NEW STATE, REWARD, and DONE
            next_state, reward, done, info = env.step(action)

            #ADJUST AND ROUND NEW STATE
            next_state = discretize(next_state)

            #UPDATE q
            if not done:
                q[state][action] = q[state][action] + LEARNING_RATE * (reward + DISCOUNT * np.amax(q[next_state]) - q[state][action])
            else:
                q[state][action] = q[state][action] + LEARNING_RATE * (reward - q[state][action])

            #ADD REWARD TO EPISODE_REWARDS
            episode_rewards += reward

            #Set state = next_state
            state = next_state

        #Decay Epsilon
        epsilon = max(epsilon * EPSILON_DECAY, 0.1)
        
        #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS
        total_episode_rewards.append(episode_rewards)
        print("EPISODE: {}    REWARD: {}    EPSILON: {}".format(episode, episode_rewards, epsilon))
