import tensorflow as tf

precision          = tf.float32
paths_eval         = 20        # number of episodes used for evaluating a policy
min_trans          = 3e3       # warmup time (number of transitions before the learning starts)
learn_trans        = 3e6       # max learning transations after warmup
max_trans          = 1e5       # max data size
eval_every         = 10000      # evaluate the current policy after X steps of learning
batch_size         = 64         # random mini-batch size for computing the gradients
lrate_v            = 1e-4       # ADAM learning rate (for learning V)
lrate_q            = 1e-4
lrate_pi           = 1e-4
gamma              = 0.99       # discount factor
tau_critic         = 0.005        # coefficient for target critic soft update
std_noise          = 2.         # Gaussian policy std
filter_env         = False      # True to normalize actions and states to [-1,1]
pi_activations     = [tf.nn.tanh, tf.nn.tanh]
v_activations      = [tf.nn.tanh, tf.nn.tanh]
q_activations      = [tf.nn.tanh, tf.nn.tanh]
q_sizes            = [64, 64]
v_sizes            = [64, 64]
pi_sizes           = [64, 64]
lrate_alpha        = 1e-4

config_env = {
    'MountainCarContinuous-v0' : {
        'std_noise'      : 10.5,
        'pi_sizes'       : [15, 45],
        'v_sizes'        : [15, 45],
        'q_sizes'        : [15, 45],
    },
    'Pendulum-v0' : {
        'std_noise'      : 4.,
        'pi_sizes'       : [15, 45],
        'v_sizes'        : [15, 45],
        'q_sizes'        : [15, 45],
        'lrate_v'        : 1e-3,
        'lrate_q'        : 1e-3,
        'lrate_pi'       : 1e-3,
    },
    'Reacher-v2' : {
        'maxiter'        : 5000,
    },
}
