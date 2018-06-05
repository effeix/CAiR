import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding


class MyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = 1

    def step(self, action):
        obs = None
        reward = None
        done = False
        info = {}

        return obs, reward, done, info

    def reset(self):
        obs = None
        return obs

    def render(self, mode='human'):
        print(1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        x = 1
