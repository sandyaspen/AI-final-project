import gym

#Barebones example from https://gym.openai.com/docs/#Environments
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()