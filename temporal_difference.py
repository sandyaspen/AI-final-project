import gym
import random
import numpy as np
import pandas as pd


EPISODES = 2500
LEARNING_RATE = 1
DISCOUNT = .83 
EPSILON = 1


class Temporal_Difference:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.cart_position_bins = pd.cut([self.env.observation_space.low[0], self.env.observation_space.high[0]], bins=5, retbins=True)[1]
        self.cart_velocity_bins = pd.cut([self.env.observation_space.low[1], self.env.observation_space.high[1]], bins=5, retbins=True)[1]
        self.pole_angle_bins = pd.cut([self.env.observation_space.low[2], self.env.observation_space.high[2]], bins=10, retbins=True)[1]
        self.pole_angular_velocity_bins = pd.cut([self.env.observation_space.low[3], self.env.observation_space.high[3]], bins=10, retbins=True)[1]
        self.episodes=EPISODES
        self.discount=DISCOUNT 
        self.learning_rate=LEARNING_RATE
        self.epsilon=EPSILON 
        self.q = np.zeros((5,5,10,10,2), dtype=float)


    def set_hyper_params(self, **kwargs):
        if 'episodes' in kwargs:
            self.episodes = kwargs['episodes']
        if 'discount' in kwargs:
            self.discount = kwargs['discount']
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']


    def bin(self, val, bins):
        for i in range(len(bins)):
            if val >= bins[i] and val < bins[i+1]:
                return i


    def discretize(self, state):
        return (self.bin(state[0], self.cart_position_bins), self.bin(state[1], self.cart_velocity_bins), self.bin(state[2], self.pole_angle_bins), self.bin(state[3], self.pole_angular_velocity_bins))


    def test(self, q=None, episodes=100):
        if not q is None:
            self.q = q
        return self.run(training=False, episodes=episodes)


    def train(self):
        return self.run(training=True, episodes=self.episodes)


    def run(self, training, episodes):
        epsilon = self.epsilon 
        learning_rate = self.learning_rate
        total_episode_rewards = []
        total_episodes = episodes
        for episode in range(episodes):
            state = self.env.reset()
            state = self.discretize(state)
            done = False
            episode_rewards = 0

            while not done:
                if training and random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q[state])
                #PERFORM ACTION & COLLECT NEW STATE, REWARD, and DONE
                next_state, reward, done, info = self.env.step(action)

                #ADJUST AND ROUND NEW STATE
                next_state = self.discretize(next_state)

                #UPDATE q
                if training:
                    if not done:
                        self.q[state][action] = self.q[state][action] + learning_rate * (reward + self.discount * np.amax(self.q[next_state]) - self.q[state][action])
                    else:
                        self.q[state][action] = self.q[state][action] + learning_rate * (reward - self.q[state][action])

                #ADD REWARD TO EPISODE_REWARDS
                episode_rewards += reward

                #Set state = next_state
                state = next_state

            #Decay Epsilon
            epsilon = max((total_episodes - (episode * 1.15)) / total_episodes, 0.05)
            learning_rate = max((total_episodes - (episode * 1.15)) / total_episodes, 0.1)
            
            
            #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS
            total_episode_rewards.append(episode_rewards)
            if training:
                print("EPISODE: {}    REWARD: {}    EPSILON: {}".format(episode, episode_rewards, epsilon), end='\r')
            else:
                print("EPISODE: {}    REWARD: {}".format(episode, episode_rewards), end='\r')
        print()
        return self.q, np.array(total_episode_rewards)


if __name__ == '__main__':
    td = Temporal_Difference()
    q, _ = td.train()
    _, episode_rewards = td.test(q)
    print(episode_rewards)
    print(np.mean(episode_rewards))

