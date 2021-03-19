import gym
import random
import numpy as np
import pandas as pd


EPISODES = 2500
LEARNING_RATE = 1
DISCOUNT = .93
EPSILON = 1


class Monte_Carlo:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.cart_position_bins = pd.cut([self.env.observation_space.low[0], self.env.observation_space.high[0]], bins=5, retbins=True)[1]
        self.cart_velocity_bins = pd.cut([self.env.observation_space.low[1], self.env.observation_space.high[1]], bins=5, retbins=True)[1]
        self.pole_angle_bins = pd.cut([self.env.observation_space.low[2], self.env.observation_space.high[2]], bins=10, retbins=True)[1]
        self.pole_angular_velocity_bins = pd.cut([self.env.observation_space.low[3], self.env.observation_space.high[3]], bins=10, retbins=True)[1]
        self.episodes=EPISODES
        self.discount=DISCOUNT 
        self.epsilon=EPSILON
        self.learning_rate = LEARNING_RATE
        self.q = np.zeros((5,5,10,10,2), dtype=float)
        self.qq = np.zeros((5,5,10,10,2,2), dtype=float)


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
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        

    def discretize(self, state):
        return (self.bin(state[0], self.cart_position_bins), 
                self.bin(state[1], self.cart_velocity_bins), 
                self.bin(state[2], self.pole_angle_bins), 
                self.bin(state[3], self.pole_angular_velocity_bins))


    def test(self, q=None, qq=None, episodes=100):
        if not q is None:
            self.q = q
        if not qq is None:
            self.qq = qq
        return self.run(training=False, episodes=episodes)


    def train(self, episodes):
        return self.run(training=True, episodes=episodes)


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
            
            episode_memory = [] # stores the (state, action, reward) tuple

            while not done:
                if training and random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q[state])

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
            if training:
                discounted_reward = 0
                for state, action, reward in episode_memory[::-1]:
                    discounted_reward = reward + (self.discount * discounted_reward)      
                    self.qq[state][action][0] += 1
                    self.qq[state][action][1] += discounted_reward
                    #self.q[state][action] = self.qq[state][action][1]/self.qq[state][action][0]
                    self.q[state][action] = self.q[state][action] + (learning_rate * (discounted_reward - self.q[state][action]))
                            
            #Decay Epsilon
            epsilon = max((total_episodes - (episode * 1.15)) / total_episodes, 0.05)
            learning_rate = max((total_episodes - (episode * 1.15)) / total_episodes, 0.1)
            
            #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS
            total_episode_rewards.append(episode_rewards)
            """
            if training:
                print("EPISODE: {}    REWARD: {}    EPSILON: {}".format(episode, episode_rewards, epsilon), end='\r')
            else:
                print("EPISODE: {}    REWARD: {}".format(episode, episode_rewards), end='\r')
        print()
            """
        return self.q, self.qq, np.array(total_episode_rewards)


if __name__ == "__main__":
    Monte_Carlo().run()
