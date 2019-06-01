import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Very simple chainwalk with simple dynamics.
The state is the agent position (x) and velocity (xd), and the bonus location (b).
The action is the acceleration.
The location of the bonus is randomly chosen at the beginning of the episode among
two possible positions: either behind the initial position of the agent, or behind the goal.
The initial position is fixed, as well as the goal position.
The agent gets the bonus is it is very close to it with almost 0 velocity. The bonus then disappears.
The episode ends when the agent is very close to the goal with almost 0 velocity.


---- HOW TO USE ----

* Place this file in gym/gym/envs/classic_control
* Add to __init__.py (located in the same folder)

    from gym.envs.classic_control.toy_chain_sparse_env import ToyChainSparseEnv

* Register the environment in your script

    gym.envs.register(
         id='ToyChainSparse-v0',
         entry_point='gym.envs.classic_control:ToyChainSparseEnv',
         max_episode_steps=1000,
    )
'''

class ToyChainSparseEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.dt = 0.1
        self.tol = 0.01
        self.goal_state = 0.8
        self.bonus_states = [-0.2, 1.]
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        u = np.clip(u, self.action_space.low, self.action_space.high)
        x = self.state[0]
        xd = self.state[1]
        b = self.state[2]
        xd_n = xd + u*self.dt
        x_n = x + xd*self.dt

        done = False
        rwd = 0.
        dist_goal = np.abs(x - self.goal_state)
        dist_bonus = np.abs(x - b)
        if dist_bonus < self.tol and np.abs(xd) < self.tol:
            rwd = 1.
            b = self.goal_state # place bonus where the goal is
        if dist_goal < self.tol and np.abs(xd) < self.tol:
            rwd = 0.5 # overwrite bonus reward with goal reward (because you already collected it)
            done = True

        self.state[0] = x_n
        self.state[1] = xd_n
        self.state[2] = b
        self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        # rwd -= 0.001*u**2
        return self._get_obs(), rwd, done, {}

    def reset(self):
        self.state = np.array([0,0,0])
        self.state[2] = self.bonus_states[np.random.randint(0,2)]
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state
