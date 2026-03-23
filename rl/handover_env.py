import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class HandoverEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # action space: choosing 1 of 4 BS
        self.action_space = Discrete(4)

        # observation Space
        low = np.array([0] * 8 + [0] * 4)  # Min values for RSRP/RSRQ and One-Hot
        high = np.array([127] * 8 + [1] * 4)  # Max values
        self.observation_space = Box(low=low, high=high, dtype=np.int32)

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
