import gym
from gym import spaces
import numpy as np

'''
---- DESCRIPTION ----

Dynamics are linear (s' = s + a) and reward is sparse.
There are multiple goals, with the furthest yielding the highest reward.
The initial position is fixed and the episode ends when a reward is collected.

With the default implementation, the highest reward is located in [20,20] and
needs 20 steps to be collected. To make it challenging, set max_episode_steps=25.


---- HOW TO USE ----

* Place this file in gym/gym/envs/classic_control
* Add to __init__.py (located in the same folder)

    from gym.envs.classic_control.toy_sparse_env import ToySparseEnv

* Register the environment in your script (select the state bound)

    gym.envs.register(
         id='ToySparse-v0',
         entry_point='gym.envs.classic_control:ToySparseEnv',
         max_episode_steps=25,
    )
'''

class ToySparseEnv(gym.Env):

    def __init__(self):
        self.size = 2 # dimensionality of state and action
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-20., high=20., shape=(self.size,), dtype=np.float32)
        self.rwd_radius = 1.
        self.rwd_states = [[1, 1], [-2, 3], [10, -2], [20, 20]]
        self.rwd_magnitude = [2, 4, 10, 50]

    def step(self,u):
        dist2 = np.sum((self.state - self.rwd_states)**2,1)
        is_close = np.where(dist2 < self.rwd_radius)[0]
        if is_close.size == 0:
            rwd = 0.
            done = False
        else:
            rwd = self.rwd_magnitude[is_close[0]]
            done = True
        rwd -= np.sum(0.01*u**2)
        self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), rwd, done, {}

    def reset(self):
        self.state = np.array([0,0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state
