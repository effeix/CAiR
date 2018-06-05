import gym
from myenv import MyEnv

env = gym.make("MyEnv-v0")

for i_episode in range(10):
    obs = env.reset()
    for t in range(1):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
