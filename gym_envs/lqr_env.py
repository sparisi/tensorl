import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Very simple linear-quadratic regulator, with dynamics
s' = As + Ba

and reward
r = - s'Qs - a'Ru

In this simple task, A, B, Q, R are all identity matrices.
The initial state is drawn from a uniform distribution in [-s0,s0].
All value functions and average return can be computed in closed form if the
policy is linear in the state, i.e., a = Ks.


---- HOW TO USE ----

* Place this file in gym/gym/envs/classic_control
* Add to __init__.py (located in the same folder)

    from gym.envs.classic_control.lqr_env import LqrEnv

* Register the environment in your script (select the size, initial state and state bound)

    gym.envs.register(
         id='Lqr-v0',
         entry_point='gym.envs.classic_control:LqrEnv',
         max_episode_steps=150,
         kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : 20.},
    )
'''

class LqrEnv(gym.Env):

    def __init__(self, size, init_state, state_bound):
        self.init_state = init_state
        self.size = size # dimensionality of state and action
        self.action_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,), dtype=np.float32)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        costs = np.sum(u**2) + np.sum(self.state**2)
        self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = self.init_state*np.ones((self.size,))
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state



    def riccati_matrix(self, K, gamma=1.):
        tolerance = 0.0001
        converged = False
        itr = 0
        maxitr = 500
        I = np.eye(self.size)
        P = I
        Pnew = I + gamma*P + gamma*np.dot(K.T,P) + gamma*np.dot(P,K) + gamma*np.dot(K.T,P).dot(K) + np.dot(K.T,K)
        while not converged or itr < maxitr:
            P = Pnew
            Pnew = I + gamma*P + gamma*np.dot(K.T,P) + gamma*np.dot(P,K) + gamma*np.dot(K.T,P).dot(K) + np.dot(K.T,K)
            P_diff = P - Pnew
            if np.any(np.isnan(P_diff)) or np.any(np.isinf(P_diff)):
                break
            converged = np.max(P_diff) < tolerance
            itr += 1
        return P

    def v_function(self, K, state, gamma=1.):
        return - np.sum(np.dot(np.square(state),self.riccati_matrix(K,gamma)), axis=1)

    def q_function(self, K, state, action, gamma=1.):
        I = np.eye(self.size)
        tmp = state + action
        return - np.sum(np.square(state) + np.square(action), axis=1, keepdims=True) - gamma*np.dot(np.square(tmp),self.riccati_matrix(K,gamma))

    def avg_return(self, K, gamma=1.):
        P = self.riccati_matrix(K,gamma)
        high = self.init_state*np.ones((self.size,))
        Sigma_s = np.diag(high+high)**2 / 12.
        return - np.trace(Sigma_s*P)
