import numpy as np
import random
import gym

import pandas as pd

EPISODES = 5000
DISCOUNT = .95
EPSILON = 1
EPSILON_DECAY = 0.997

class Monte_Carlo:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.cart_position_bins = pd.cut([self.env.observation_space.low[0], self.env.observation_space.high[0]], bins=5, retbins=True)[1]
        self.cart_velocity_bins = pd.cut([self.env.observation_space.low[1], self.env.observation_space.high[1]], bins=5, retbins=True)[1]
        self.pole_angle_bins = pd.cut([self.env.observation_space.low[2], self.env.observation_space.high[2]], bins=10, retbins=True)[1]
        self.pole_angular_velocity_bins = pd.cut([self.env.observation_space.low[3], self.env.observation_space.high[3]], bins=10, retbins=True)[1]
        self.episodes=EPISODES
        self.discount=DISCOUNT 
        self.epsilon=EPSILON
        self.epsilon_decay=EPSILON_DECAY

    def bin(self, val, bins):
        for i in range(len(bins)):
            if val >= bins[i] and val < bins[i+1]:
                return i
    
    def set_hyper_params(self, **kwargs):
        if 'episodes' in kwargs:
            self.episodes = kwargs['episodes']
        if 'discount' in kwargs:
            self.discount = kwargs['discount']
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        if 'epsilon_decay' in kwargs:
            self.epsilon_decay = kwargs['epsilon_decay']
        
    def discretize(self, state):
        return (self.bin(state[0], self.cart_position_bins), 
                self.bin(state[1], self.cart_velocity_bins), 
                self.bin(state[2], self.pole_angle_bins), 
                self.bin(state[3], self.pole_angular_velocity_bins))

    def run(self):
        q = np.zeros((5,5,10,10,2), dtype=float)
        qq = np.zeros((5,5,10,10,2,2), dtype=float)
        for episode in range(self.episodes):
            state = self.env.reset()
            state = self.discretize(state)
            done = False
            episode_rewards = 0
            
            episode_memory = [] # stores the (state, action, reward) tuple

            while not done:
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(q[state])

                #PERFORM ACTION & COLLECT NEW STATE, REWARD, and DONE
                next_state, reward, done, info = self.env.step(action)
                
                #ADJUST AND ROUND NEW STATE
                next_state = self.discretize(next_state)
                
                #UPDATE
                episode_memory.append((state, action, reward))
                
                #ADD REWARD TO EPISODE_REWARDS
                episode_rewards += reward
                
                #set state to next_state
                state = next_state
            
            
            # calculate discounted_reward based by iterating backward through `episode_memory_sar`
            discounted_reward = 0
            for state, action, reward in episode_memory[::-1]:
                discounted_reward = reward + (self.discount * discounted_reward)      
                qq[state][action][0] += 1
                qq[state][action][1] += discounted_reward
                q[state][action] = qq[state][action][1]/qq[state][action][0]
                #q[state][action] = q[state][action] + (LEARNING_RATE * (discounted_reward - q[state][action]))
                        
            #Decay Epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.1)
            
            #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS
            print("EPISODE: {}    REWARD: {}    EPSILON: {}".format(episode, episode_rewards, self.epsilon), end='\r')


if __name__ == "__main__":
    Monte_Carlo().run()