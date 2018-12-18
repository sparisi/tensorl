import tensorflow as tf

precision          = tf.float32
verbose            = True       # True to show info about REPS inner optimization
maxiter            = 5000         # number of learning iterations
min_trans_per_iter = 10000       # minimum number of transition steps per iteration (if an episode ends before min_trans_per_iter is reached, a new one starts)
max_trans          = 1000      # max number of steps per episode during exploration
paths_eval         = 100        # number of episodes used for evaluating a policy
max_reuse          = 1          # the algorithm reuses the samples from the last X-1 iterations
epsilon            = 0.1       # KL bound
filter_env         = False      # True to normalize actions and states to [-1,1]
reset_prob         = 0.01       # reset probabiliy for infinite horizon
std_noise          = 2.        # Gaussian policy std
n_fourier          = 500       # number of Fourier features

config_env = {
    'MountainCarContinuous-v0' : {
        'std_noise'      : 10.5,
        'max_trans'      : 3000,
        'reset_prob'     : 0.01,
        'n_fourier'      : 50,
    },
    'Swimmer-v2' : {
        'reset_prob'     : 0.05,
        'max_trans'      : 3000,
        'n_fourier'      : 100,
    },
    'Pendulum-v0' : {
        'reset_prob'     : 0.05,
        'max_trans'      : 3000,
        'std_noise'      : 4.,
        'n_fourier'      : 75,
        'maxiter'        : 50,
    },
    'Reacher-v2' : {
        'maxiter'        : 2500,
    },
    'HalfCheetah-v2' : {
        'reset_prob'     : 0.01,
        'epsilon'        : 0.01,
        'n_fourier'      : 250,
    },
}
