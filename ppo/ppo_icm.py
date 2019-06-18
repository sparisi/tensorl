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
    nobs = tf.placeholder(dtype=precision, shape=[None, obs_size], name='nobs')

    # Build pi
    act_bound = env.action_space.high
    assert np.all(act_bound == -env.action_space.low)
    mean = MLP([obs], pi_sizes+[act_size], pi_activations+[None], 'pi_mean')
    with tf.variable_scope('pi_std'): std = tf.Variable(std_noise * tf.ones([1, act_size], dtype=precision), dtype=precision)
    pi = MVNPolicy(session, obs, mean.output[0], std, act_bound=act_bound)

    # Build forward dynamics
    fwd = MLP([tf.concat([obs,pi.act],axis=1)], fwd_sizes+[obs_size], fwd_activations+[None], 'fwd')

    # Build V
    v = MLP([obs], v_sizes+[1], v_activations+[None], 'v')

    # V optimization
    target_v = tf.placeholder(dtype=precision, shape=[None, 1], name='target_v')
    loss_v = tf.losses.mean_squared_error(v.output[0], target_v)
    optimize_v = tf.train.AdamOptimizer(lrate_v).minimize(loss_v)

    # pi optimization
    advantage = tf.placeholder(dtype=precision, shape=[None, 1], name='advantage')
    old_log_probs = tf.placeholder(dtype=precision, shape=[None, 1], name='old_log_probs')
    prob_ratio = tf.exp(pi.log_prob - old_log_probs)
    clip_pr = tf.clip_by_value(prob_ratio, 1.-e_clip, 1.+e_clip)
    loss_pi = -tf.reduce_mean(tf.minimum(tf.multiply(prob_ratio, advantage), tf.multiply(clip_pr, advantage)))
    optimize_pi = tf.train.AdamOptimizer(lrate_pi).minimize(loss_pi)

    # Forward dynamics optimization
    loss_fwd = tf.losses.mean_squared_error(fwd.output[0], nobs)
    optimize_fwd = tf.train.AdamOptimizer(lrate_fwd).minimize(loss_fwd)

    # Define curiosity
    curiosity = tf.reduce_mean(tf.square(fwd.output[0] - nobs), axis=1)

    # Init variables
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    v.reset(session, 0.)

    logger = LoggerData('ppo_icm', env_name, run_name)
    print()
    print('    V LOSS                         PI LOSS                        ENTROPY        RETURN')
    for itr in range(maxiter):
        paths = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter)
        dct_model = {obs: paths["obs"], pi.act: paths["act"], nobs: paths["nobs"]}
        nb_trans = len(paths["rwd"])

        # Update dynamics models
        for epoch in range(epochs_fwd):
            if epoch == 0:
                fwd_loss_before = session.run(loss_fwd, dct_model)
            for batch_idx in minibatch_idx_list(batch_size, len(paths["rwd"])):
                session.run(optimize_fwd, {obs: paths["obs"][batch_idx], pi.act: paths["act"][batch_idx], nobs: paths["nobs"][batch_idx]})
        fwd_loss_after = session.run(loss_fwd, dct_model)

        # Update V
        for epoch in range(epochs_v):
            v_values = session.run(v.output[0], {obs: paths["obs"]})
            a_values = gae(paths, v_values, gamma, lambda_trace) # compute the advantage
            target_values = v_values + a_values # generalized Bellman operator
            if epoch == 0:
                v_loss_before = session.run(loss_v, {obs: paths["obs"], target_v: target_values})
            for batch_idx in minibatch_idx_list(batch_size, nb_trans):
                session.run(optimize_v, {obs: paths["obs"][batch_idx], target_v: target_values[batch_idx]})
        v_loss_after = session.run(loss_v, {obs: paths["obs"], target_v: target_values})

        # Estimate advantages and intrinsic curiosity (ic)
        v_values = session.run(v.output[0], {obs: paths["obs"]})
        a_values = gae(paths, v_values, gamma, lambda_trace)
        a_values = (a_values - np.mean(a_values)) / np.std(a_values)
        ic_values = np.asarray(session.run(curiosity, dct_model))[:,None]
        ic_values = (ic_values - np.mean(ic_values)) / np.std(ic_values)

        a_values = a_values + beta * ic_values # add curiosity
        a_values = (a_values - np.mean(a_values)) / np.std(a_values)

        # Udpate pi
        old_lp = pi.get_log_prob(paths["obs"], paths["act"])
        pi_loss_before = session.run(loss_pi, {obs: paths["obs"], pi.act: paths["act"], old_log_probs: old_lp, advantage: a_values})
        for epoch in range(epochs_pi):
            for batch_idx in minibatch_idx_list(batch_size, nb_trans):
                dct_pi = {obs: paths["obs"][batch_idx],
                            pi.act: paths["act"][batch_idx],
                            old_log_probs: old_lp[batch_idx],
                            advantage: a_values[batch_idx]}
                session.run(optimize_pi, dct_pi)
        pi_loss_after = session.run(loss_pi, {obs: paths["obs"], pi.act: paths["act"], old_log_probs: old_lp, advantage: a_values})

        # Evaluate pi
        avg_rwd = np.sum(paths["rwd"]) / paths["nb_paths"]
        entr = pi.estimate_entropy(paths["obs"])
        print('%d | %e -> %e   %e -> %e   %e   %e   ' % (itr, v_loss_before, v_loss_after, pi_loss_before, pi_loss_after, entr, avg_rwd), flush=True)
        with open(logger.fullname, 'ab') as f:
            np.savetxt(f, np.atleast_2d([fwd_loss_before, fwd_loss_after, v_loss_before, v_loss_after, pi_loss_before, pi_loss_after, entr, avg_rwd])) # save data

    session.close()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
