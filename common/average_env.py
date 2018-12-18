import numpy as np

def make_average_env(env, reset_prob=0.):
    '''
    In RL, we typically consider MDPs with discounted rewards. This function
    switches to the average reward setting by introducing state resets, i.e.,
    the environment resets itself to its (random) initial state with some
    probability. To guarantee the Markovian property of the MDP, we cannot have
    time-dependent resets. However, the reset can depend on the current state.
    Some possibilities are:
    1) Always terminating only in terminal states,
    2) Always terminating in terminal states, and with fixed probabiliy elsewhere,
    3) Always terminating with fixed probability everywhere,
    4) Terminating with a state-dependent probability.

    The first case is the default gym implementation. This function implements
    the second case.

    In theory, there should be no horizon limit (_max_episode_steps) and an
    episode should end only according to terminal states and resets.
    You can change this by commenting some lines in the `step` function below.

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
            done = done or (self._max_episode_steps <= self._elapsed_steps) or (np.random.rand() > (1.-self.reset_prob))
            # done = np.random.rand() > (1.-self.reset_prob)
            return obs, rwd, done, info

    average_env = AverageEnv()

    print()
    print('Switching to the average reward setting.')
    print('Reset probability {:f}.'.format(reset_prob))
    print()

    return average_env
