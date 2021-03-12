import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import sys
import pickle
from os import path


EPISODES = 5000
LEARNING_RATE = 0.2
DISCOUNT = 1 
EPSILON = 1
EPSILON_DECAY = 0.999


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        if path.exists("td_q_matrix.pickle"):
            with open("td_q_matrix.pickle", "rb") as f:
                q = pickle.load(f)
    else:
        q = np.zeros((18,15,3), dtype=float)

    env = gym.make('MountainCar-v0')
    total_episode_rewards = []
    epsilon = EPSILON
    for episode in range(EPISODES):
        state = env.reset()
        state = (int(round(state[0] - env.observation_space.low[0], 1) * 10), int(round(state[1] - env.observation_space.low[1], 2) * 100))
        done = False
        episode_rewards = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])
            #PERFORM ACTION
            #COLLECT NEW STATE, REWARD, and DONE
            next_state, reward, done, info = env.step(action)

            #ADJUST AND ROUND NEW STATE
            next_state = (int(round(next_state[0] - env.observation_space.low[0], 1) * 10), int(round(next_state[1] - env.observation_space.low[1], 2) * 100))

            #UPDATE q
            if not done:
                q[state][action] = q[state][action] + LEARNING_RATE * (reward + DISCOUNT * np.amax(q[next_state]) - q[state][action])
            else:
                q[state][action] = q[state][action] + LEARNING_RATE * (reward - q[state][action])

            #ADD REWARD TO EPISODE_REWARDS
            episode_rewards += reward

            #Set state = next_state
            state = next_state

        #Lower epsilon some amount or check to lower epsilon if episode % SOME_VALUE == 0
        epsilon *= EPSILON_DECAY
        
        #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS
        total_episode_rewards.append(episode_rewards)
        print("EPISODE: {}    REWARD: {}    EPSILON: {}".format(episode, episode_rewards, epsilon))


    #Save the trained q matrix 
    with open("td_q_matrix.pickle", "wb") as f:
        pickle.dump(q, f)


    #Plot results with matplot lib
    #Only plot every 50th episode
    to_plot = total_episode_rewards[0:EPISODES:50]
    fig, ax = plt.subplots()
    ax.plot(range(0,EPISODES,50), to_plot)
    ax.set(xlabel='Episode number', ylabel='Rewards recieved during episode',
        title="Temporal Difference Learning Results")
    ax.grid()
    fig.savefig("TDresults.png")
    plt.show()
