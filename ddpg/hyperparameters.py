import tensorflow as tf

precision      = tf.float64
min_trans      = 5e3       # warmup time (number of transitions before the learning starts)
learn_trans    = 1e6       # max learning transations after warmup
max_trans      = 5e3       # max data size
eval_every     = 1000      # evaluate the current policy after X steps of learning
paths_eval     = 20       # number of episodes used for evaluating a policy
batch_size     = 64        # random mini-batch size for computing the gradients
gamma          = 0.99      # discount factor
std_noise      = 0.2       # noise of the OU process
tau_q          = 0.001      # soft update of the target Q network
tau_pi         = 0.001      # soft update of the target pi network
lrate_pi       = 1e-3      # ADAM learning rate (policy optimizer)
lrate_q        = 1e-3      # ADAM learning rate (Q network optimizer)
filter_env     = False     # True to normalize actions and states to [-1,1]
q_activations  = [tf.nn.tanh, tf.nn.tanh]
pi_activations = [tf.nn.tanh, tf.nn.tanh]
pi_sizes       = [64, 64]
q_sizes        = [64, 64]

alpha_init     = 0.1         # TD-regularization coefficient
alpha_decay    = 0.99999     # alpha = alpha*alpha_decay

config_env = {
    'Pendulum-v0' : {
        'std_noise'      : 16.,
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [15, 45],
    },
    'MountainCarContinuous-v0' : {
        'std_noise'      : 2.,
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [15, 45],
    },
    'RoboschoolInvertedPendulum-v1' : {
        'batch_size'     : 64,
        'learn_trans'    : 15e4,
        'max_trans'      : 5e4,
        'lrate_pi'       : 1e-4,
        'lrate_q'        : 1e-3,
    },
    'RoboschoolHalfCheetah-v1' : {
        'max_trans'      : 5e4,
        'learn_trans'    : 1e6,
        'pi_sizes'       : [200, 200],
        'q_sizes'        : [200, 200],
    },
    'InvertedPendulumSwingupBulletEnv-v0' : {
        'pi_sizes'       : [16, 16],
        'q_sizes'        : [16, 16],
    },
}
