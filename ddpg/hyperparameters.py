import tensorflow as tf

precision      = tf.float32
min_trans      = 1e4       # warmup time (number of transitions before the learning starts)
learn_trans    = 10e6      # max learning transations after warmup
max_trans      = 3e5       # max data size
eval_every     = 1000      # evaluate the current policy after X steps of learning
paths_eval     = 200       # number of episodes used for evaluating a policy
batch_size     = 64        # random mini-batch size for computing the gradients
gamma          = 0.99      # discount factor
std_noise      = 0.1       # policy noise
tau_q          = 0.005     # soft update of the target Q network
tau_pi         = 0.005     # soft update of the target pi network
lrate_pi       = 1e-4      # ADAM learning rate (policy optimizer)
lrate_q        = 1e-4      # ADAM learning rate (Q network optimizer)
filter_env     = False     # True to normalize actions and states to [-1,1]
q_activations  = [tf.nn.tanh, tf.nn.tanh]
pi_activations = [tf.nn.tanh, tf.nn.tanh]
pi_sizes       = [64, 64]
q_sizes        = [64, 64]

alpha_init     = 0.1         # TD-regularization coefficient
alpha_decay    = 0.99999     # alpha = alpha*alpha_decay

config_env = {
    'MountainCarContinuous-v0' : {
        'std_noise'      : 10.5,
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [15, 45],
    },
    'Pendulum-v0' : {
        'std_noise'      : 4.,
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [15, 45],
        'lrate_q'        : 1e-3,
        'lrate_pi'       : 1e-3,
    },
    'Reacher-v2' : {
        'learn_trans'    : 1e6,
    },
}
