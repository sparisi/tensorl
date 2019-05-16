import numpy as np

def make_average_env(env, reset_prob):
    '''
    In RL, we typically consider MDPs with discounted rewards. This function
    switches to the average reward setting by introducing state resets, i.e.,
    the environment resets itself to its (random) initial state. The reset
    probability is an argument of this function.
    To guarantee the Markovian property of the MDP, we cannot have
    time-dependent resets. However, the reset can depend on the current state.
    Some possibilities are:
    1) Terminate only in terminal states (defaul gym behavior),
    2) Terminate in terminal states, and with fixed probability elsewhere (this function),
    3) Terminate with fixed probability everywhere (i.e., ignore terminal states),
    4) Terminate with a state-dependent probability.

    In theory, there should be no horizon limit (env._max_episode_steps) and an
    episode should end only according to terminal states and resets.
    You can change this by commenting out line 32.

    Reference:
    van Hoof et al, "Non-parametric Policy Search with Limited Information Loss",
    JMLR, 2017
    '''

    env_type = type(env)

    class AverageEnv(env_type):
        def __init__(self):
            self.__dict__.update(env.__dict__) # Transfer properties
            self.reset_prob = reset_prob
            self._max_episode_steps = np.inf

        def step(self, action):
            obs, rwd, done, info = env_type.step(self, action) # Super function
            done = done or (self._max_episode_steps <= self._elapsed_steps) or (np.random.rand() < self.reset_prob)
            return obs, rwd, done, info

    average_env = AverageEnv()

    print()
    print('Switching to the average reward setting.')
    print('Reset probability {:f}.'.format(reset_prob))
    print()

    return average_env
