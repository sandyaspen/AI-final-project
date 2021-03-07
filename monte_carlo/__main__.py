import numpy as np
import random
import gym
import sys
import pickle
from os import path


EPISODES = 5000
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EPSILON = 1
EPSILON_DECAY = 0.999

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        if path.exists("mc_q_matrix.pickle"):
            with open("mc_q_matrix.pickle", "rb") as f:
                q = pickle.load(f)
        if path.exists("td_qq_matrix.pickle"):
            with open("td_qq_matrix.pickle", "rb") as f:
                q = pickle.load(f)
    else:
        q = np.zeros((18,15,3), dtype=float)
        qq = np.zeros((18,15,3,2), dtype=float)

    env = gym.make('MountainCar-v0')
    total_episode_rewards = []
    epsilon = EPSILON
    alpha = LEARNING_RATE
    gamma = DISCOUNT
    for episode in range(EPISODES):
        state = env.reset()
        state = (int(round(state[0] - env.observation_space.low[0], 1) * 10), int(round(state[1] - env.observation_space.low[1], 2) * 100))
        done = False
        episode_rewards = 0
        
        episode_memory_sar = [] # stores the (state, action, reward) tuple

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
            
            #UPDATE
            episode_memory_sar.append((state, action, reward))
            
            # increment how many times we have been in this state/action ????
            # qq[state][action][0] += 1
            
            # increase total rewards we have received in this state/action ????
            # qq[state][action][1] += reward
            
            #ADD REWARD TO EPISODE_REWARDS
            episode_rewards += reward
            
            #set state to next_state
            state = next_state
        
        
        # calculate Qt based by iterating backward through `episode_memory_sar`
        first = True
        episode_memory_saq = [] # stores the (state, action, estimated_reward) tuple
        est_reward = 0
        for state, action, reward in reversed(episode_memory_sar):
            if first:
                first = False
            else:
                # q[state][action] = q[state][action] + alpha * (Gt - q[state][action])
                qq[state][action][0] += 1
                qq[state][action][1] += est_reward
                episode_memory_saq.append((state, action, est_reward))    
            est_reward = reward + (gamma * est_reward)    
        episode_memory_saq.reverse()  
          
        # update q table based on values in `episode_memory_saq` and qq table
        for state, action, est_reward in episode_memory_saq:
            Gt = qq[state][action][1]/qq[state][action][0]
            q[state][action] = q[state][action] + alpha * (Gt - q[state][action])
            
        
        
        #Lower epsilon some amount
        epsilon *= EPSILON_DECAY
        
        #ADD EPISODE REWARD TO TOTAL_EPISODE_REWARDS
        total_episode_rewards.append(episode_rewards)
        print("EPISODE: {}    REWARD: {}    EPSILON: {}".format(episode, episode_rewards, epsilon))


    #Save the trained q matrix 
    with open("mc_q_matrix.pickle", "wb") as f:
        pickle.dump(q, f)
        
    #Save the trained qq matrix 
    with open("mc_qq_matrix.pickle", "wb") as f:
        pickle.dump(qq, f)


    #Plot results with matplot lib
    print(total_episode_rewards)
