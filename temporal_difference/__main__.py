import numpy as np
import random
import gym
import sys
import pickle
from os import path


EPISODES = 500
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EPSILON = 1
EPSILON_DECAY = 0.97


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        if path.exists("td_q_matrix.pickle"):
            with open("td_q_matrix.pickel", "rb") as f:
                q = pickle.load(f)
    else:
        q = np.zeros((14,15,3), dtype=float)

    env = gym.make('MountainCar-v0')
    total_episode_rewards = []
    epsilon = EPSILON
    for episode in range(EPISODES):
        state = env.reset()
        state = (round(state[0] - env.observation_space.low[0], 1), round(state[1] - env.observation_space.low[1], 2))
        done = False
        episode_rewards = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])
            #PERFORM ACTION
            #COLLECT NEW STATE, REWARD, and DONE
            #ADJUST AND ROUND NEW STATE
            #UPDATE q
            #ADD REWARD TO EPISODE_REWARDS
        
        #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS


    #Save the trained q matrix 
    with open("td_q_matrix.pickle", "wb") as f:
        pickle.dump(q, f)
