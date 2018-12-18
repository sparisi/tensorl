import tensorflow as tf

precision          = tf.float64
maxiter            = 100       # number of learning iterations
min_trans_per_iter = 3000      # minimum number of transition steps per iteration (if an episode ends before min_trans_per_iter is reached, a new one starts)
paths_eval         = 100       # number of episodes used for evaluating a policy
eval_every         = 1         # evaluate the current policy after X steps of learning
batch_size         = 64        # random mini-batch size for computing the gradients
max_reuse          = 1         # the algorithm reuses the samples from the last X-1 iterations
gamma              = 0.99      # discount factor
std_noise          = 2.0       # Gaussian std for exploration
lrate_pi           = 1e-3      # ADAM learning rate for learning pi
lrate_q            = 1e-3      # ADAM learning rate for learning Q
epochs_pi          = 20
epochs_q           = 20
batch_size         = 64        # random mini-batch size for computing the gradients
filter_env         = False     # True to normalize actions and states to [-1,1]
pi_activations     = [tf.nn.tanh, tf.nn.tanh]
q_activations      = [tf.nn.tanh, tf.nn.tanh]
q_sizes            = [64, 64]
pi_sizes           = [64, 64]


config_env = {
    'Pendulum-v0' : {
        'std_noise'      : 4.0,
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [45, 45],
    },
    'MountainCarContinuous-v0' : {
        'std_noise'      : 5.0,
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [15, 45],
    },
    'RoboschoolInvertedPendulum-v1' : {
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [45, 45],
    },
    'RoboschoolInvertedPendulumSwingup-v1' : {
        'pi_sizes'       : [15, 45],
        'q_sizes'        : [45, 45],
    },
}
