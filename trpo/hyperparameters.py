import tensorflow as tf

precision          = tf.float32
maxiter            = 10000       # number of learning iterations
min_trans_per_iter = 3000       # minimum number of transition steps per iteration (if an episode ends before min_trans_per_iter is reached, a new one starts)
paths_eval         = 20        # number of episodes used for evaluating a policy
batch_size         = 64         # random mini-batch size for computing the gradients
lrate_v            = 1e-4       # ADAM learning rate (for learning V)
epochs_v           = 20         # epochs of gradient descent (each epoch covers the whole dataset, divided in mini-batches)
gamma              = 0.99       # discount factor
lambda_trace       = 0.95       # coefficient for generalized advantage
kl_bound           = 0.005
cg_damping         = 1e-1       # conjugate gradient damping (for the diagonal entries)
std_noise          = 2.         # Gaussian policy std
filter_env         = False      # True to normalize actions and states to [-1,1]
pi_activations     = [tf.nn.tanh, tf.nn.tanh]
v_activations      = [tf.nn.tanh, tf.nn.tanh]
v_sizes            = [64, 64]
pi_sizes           = [64, 64]

config_env = {
    'MountainCarContinuous-v0' : {
        'std_noise'      : 10.5,
        'pi_sizes'       : [15, 45],
        'v_sizes'        : [15, 45],
    },
    'Pendulum-v0' : {
        'std_noise'      : 4.,
        'pi_sizes'       : [15, 45],
        'v_sizes'        : [15, 45],
        'lrate_v'        : 1e-3,
        'lrate_pi'       : 1e-3,
    },
    'Reacher-v2' : {
        'maxiter'        : 5000,
    },
}
