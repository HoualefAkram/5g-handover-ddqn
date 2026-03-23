import gymnasium as gym


class HandoverEnv(gym.Env):

    def __init__(self):
        super().__init__()

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
