import gymnasium as gym
from gymnasium.spaces import Discrete


class HandoverEnv(gym.Env):

    def __init__(self):
        super().__init__()
        # RSRP1, RSRP2, RSRP3, RSRP4, RSRQ1, RSRQ2, RSRQ3, RSRQ4 , Serving BS (OneHotEncoded)
        self.observation_space = Discrete(12)
        # BS1, BS2, BS3, BS4
        self.action_space = Discrete(4)

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
