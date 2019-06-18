'''
This is like REINFORCE, but instead of using Monte-Carlo estimates for Q, we use function approximation.
The Q-function is learned by minimizing the mean squared TD error.
Monte-Carlo estimates are used as Q-targets.
'''

try:
    import roboschool
except ImportError:
    pass
try:
    import pybullet_envs
except ImportError:
    pass
import gym, gym.spaces
import tensorflow as tf
import numpy as np
import sys

from common import *
from .hyperparameters import *

def main(env_name, seed=1, run_name=None):
    # Read hyperparameters
    try:
        globals().update(config_env[env_name])
    except KeyError as e:
        print()
        print('\033[93m No hyperparameters defined for \"' + env_name + '\". Using default ones.\033[0m')
        print()
        pass

    # Init environment
    env = gym.make(env_name)
    if filter_env:
        env = make_filtered_env(env)

    # Init seeds
    seed = int(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth=True
    session = tf.Session(config=config_tf)

    # Init placeholders
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    obs = tf.placeholder(dtype=precision, shape=[None, obs_size], name='obs')
    act = tf.placeholder(dtype=precision, shape=[None, act_size], name='act')
    targets_q = tf.placeholder(dtype=precision, shape=[None, 1], name='targets_q')
    targets_pi = tf.placeholder(dtype=precision, shape=[None, 1], name='targets_pi')

    # Build Q
    q = MLP([tf.concat([obs, act], axis=1)], q_sizes+[1], q_activations+[None], 'q')
    loss_q = tf.reduce_mean(0.5*tf.square(q.output[0] - targets_q))
    optimize_q = tf.train.AdamOptimizer(lrate_q).minimize(loss_q, var_list=q.vars)

    # Build pi
    act_bound = env.action_space.high
    assert np.all(act_bound == -env.action_space.low)
    mean = MLP([obs], pi_sizes+[act_size], pi_activations+[None], 'pi_mean')
    with tf.variable_scope('pi_std'): std = tf.Variable(std_noise * tf.ones([1, act_size], dtype=precision), dtype=precision)
    pi = MVNPolicy(session, obs, mean.output[0], std, act_bound=act_bound)
    loss_pi = -tf.reduce_mean(tf.multiply(tf.exp(pi.log_prob), (targets_pi - tf.reduce_mean(targets_pi)))) # simple baseline
    optimize_pi = tf.train.AdamOptimizer(lrate_pi).minimize(loss_pi, var_list=mean.vars+[std])

    # Init variables
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.) # reset pi to have almost-0 output
    q.reset(session, 0.) # reset Q to have almost-0 output


    all_paths = []
    logger = LoggerData('ac_mc', env_name, run_name)
    for itr in range(maxiter):
        # Samples for learning
        paths_iter = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter)
        all_paths.append(paths_iter)
        if len(all_paths) > max_reuse:
            del all_paths[0]
        paths = merge_paths(all_paths)

        # Estimate returns
        R = mc_ret(paths,gamma)

        # Update pi
        q_values = session.run(q.output[0], {obs: paths["obs"], act: paths["act"]})
        for epoch in range(epochs_pi):
            for batch_idx in minibatch_idx_list(batch_size, len(paths["rwd"])):
                session.run(optimize_pi, {obs: paths["obs"][batch_idx,:], pi.act: paths["act"][batch_idx,:], targets_pi: q_values[batch_idx]})

        # Update Q
        for epoch in range(epochs_q):
            for batch_idx in minibatch_idx_list(batch_size, len(paths["rwd"])):
                session.run(optimize_q, {obs: paths["obs"][batch_idx,:], act: paths["act"][batch_idx,:], targets_q: R[batch_idx]})

        # Evaluation
        if itr % eval_every == 0:
            # layers_m = session.run(mean.vars)
            # draw_fast = lambda x : fast_policy(x, layers_m)
            # avg_rwd = evaluate_policy(env, policy=pi.draw_action_det, min_paths=paths_eval)
            avg_rwd = np.sum(paths["rwd"]) / paths["nb_paths"]
            mstde = session.run(loss_q, {obs: paths["obs"], act: paths["act"], targets_q: R})
            entr = pi.estimate_entropy(paths["obs"])
            print('%d   %.4f   %.4f   %.e' % (itr, avg_rwd, entr, mstde), flush=True)
            with open(logger.fullname, 'ab') as f:
                np.savetxt(f, np.atleast_2d([avg_rwd, entr, mstde])) # save data

    session.close()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
