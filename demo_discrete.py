'''
Example to show how to use the discrete softmax policy.
This script implements PPO, but can be extended to any algorithm.
The key points are:
* Line 26: act_size = env.action_space.n (instead of env.action_space.shape[0]),
* Line 30-31: definition of the policy,
* Line 62: set clip_act=False.
So if you want to solve a task with discrete actions, just modify the main script
of any other algorithm in this repository accordingly.
'''

import gym, gym.spaces
import tensorflow as tf
import numpy as np

from common import *

def main():
    session = tf.Session()

    # Init environment
    env = gym.make('CartPole-v0')

    # Init placeholders
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    obs = tf.placeholder(dtype=tf.float32, shape=[None, obs_size], name='obs')

    # Build pi
    f = MLP([obs], [64, 64, act_size], [tf.nn.tanh, tf.nn.tanh, None], 'f')
    pi = SoftmaxPolicy(session, obs, f.output[0], act_size)

    # Build V
    v = MLP([obs], [64, 64, 1], [tf.nn.tanh, tf.nn.tanh, None], 'v')

    # V optimization
    target_v = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target_v')
    loss_v = tf.losses.mean_squared_error(v.output[0], target_v)
    optimize_v = tf.train.AdamOptimizer(1e-4).minimize(loss_v)

    # pi optimization
    e_clip = 0.05
    gamma = 0.99
    lambda_trace = 0.95
    advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='advantage')
    old_log_probs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_log_probs')
    prob_ratio = tf.exp(pi.log_prob - old_log_probs)
    clip_pr = tf.clip_by_value(prob_ratio, 1.-e_clip, 1.+e_clip)
    loss_pi = -tf.reduce_mean(tf.minimum(prob_ratio*advantage, clip_pr*advantage))
    optimize_pi = tf.train.AdamOptimizer(1e-4).minimize(loss_pi)

    # Init variables (better not to initialize f to 0)
    session.run(tf.global_variables_initializer())
    v.reset(session, 0.)
    f.reset(session, 0.)

    epochs_v = 20
    epochs_pi = 20
    batch_size = 64

    print()
    for itr in range(250):
        paths = collect_samples(env, policy=pi.draw_action, min_trans=3000, clip_act=False)
        nb_trans = len(paths["rwd"])

        # Update V
        for epoch in range(epochs_v):
            v_values = session.run(v.output[0], {obs: paths["obs"]})
            a_values = gae(paths, v_values, gamma, lambda_trace) # compute the advantage
            target_values = v_values + a_values # generalized Bellman operator
            for batch_idx in minibatch_idx_list(batch_size, nb_trans):
                session.run(optimize_v, {obs: paths["obs"][batch_idx], target_v: target_values[batch_idx]})

        # Estimate advantages and TD error
        v_values = session.run(v.output[0], {obs: paths["obs"]})
        a_values = gae(paths, v_values, gamma, lambda_trace)
        td_values = gae(paths, v_values, gamma, 0)

        # Standardize advantages
        a_values = (a_values - np.mean(a_values)) / np.std(a_values)

        # Udpate pi
        old_lp = pi.get_log_prob(paths["obs"], paths["act"])
        for epoch in range(epochs_pi):
            for batch_idx in minibatch_idx_list(batch_size, nb_trans):
                dct_pi = {obs: paths["obs"][batch_idx],
                            pi.act: paths["act"][batch_idx],
                            old_log_probs: old_lp[batch_idx],
                            advantage: a_values[batch_idx]}
                session.run(optimize_pi, dct_pi)

        avg_rwd = np.sum(paths["rwd"]) / paths["nb_paths"]
        entr = pi.estimate_entropy(paths["obs"])
        print('%d | %e  %e   ' % (itr, avg_rwd, entr), flush=True)

    session.close()



if __name__ == '__main__':
    main()
