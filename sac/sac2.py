'''
SAC in PPO/TRPO style, i.e., the policy is updated after full episodes are collected.
See also https://arxiv.org/pdf/1901.11275.pdf (section 4.1).
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
    obs = tf.placeholder(dtype=precision, shape=[None, obs_size])
    nobs = tf.placeholder(dtype=precision, shape=[None, obs_size])
    act = tf.placeholder(dtype=precision, shape=[None, act_size])
    rwd = tf.placeholder(dtype=precision, shape=[None, 1])
    done = tf.placeholder(dtype=precision, shape=[None, 1])

    # Build pi
    act_bound = np.asscalar(env.action_space.high[0])
    assert act_bound == -np.asscalar(env.action_space.low[0])
    mean = MLP([obs], pi_sizes+[act_size], pi_activations+[None], 'pi_mean')
    with tf.variable_scope('pi_std'): std = tf.Variable(std_noise * tf.ones([1, act_size], dtype=precision), dtype=precision)
    pi = MVNPolicy(session, obs, mean.output[0], std, act_bound=act_bound)

    # Build V and Q
    v = MLP([obs], v_sizes+[1], v_activations+[None], 'v')
    v_t = MLP([nobs], v_sizes+[1], v_activations+[None], 'target_v')
    q = MLP([tf.concat([obs, act], axis=1), tf.concat([obs, pi.output], axis=1)], q_sizes+[1], q_activations+[None], 'q')

    # V and Q optimization
    alpha = tf.placeholder(dtype=precision)
    loss_v = tf.losses.mean_squared_error(v.output[0], q.output[1] - alpha*pi.log_prob)
    loss_q = tf.losses.mean_squared_error(q.output[0], rwd + gamma*v_t.output[0]*(1.-done))
    optimize_v = tf.train.AdamOptimizer(lrate_v).minimize(loss_v, var_list=v.vars)
    optimize_q = tf.train.AdamOptimizer(lrate_q).minimize(loss_q, var_list=q.vars)

    # pi optimization
    loss_pi = tf.reduce_mean(tf.exp(pi.log_prob) * (pi.log_prob - (q.output[1] - v.output[0]))) # V(s) as baseline
    optimize_pi = tf.train.AdamOptimizer(lrate_pi).minimize(loss_pi, var_list=mean.vars+[std])

    # Init variables
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    v.reset(session, 0.)
    q.reset(session, 0.)

    # Update target network op
    update_vt = []
    for vars_v, vars_vt in zip(v.vars, v_t.vars):
        session.run(tf.assign(vars_vt, vars_v))
        update_vt.append(tf.assign(vars_vt, tau_v*vars_v + (1.-tau_v)*vars_vt)) # soft target update

    logger = LoggerData('sac2', env_name, run_name)
    print()
    print('    ENTROPY        KL             RETURN          MSTDE')
    for itr in range(maxiter):
        paths = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter)

        old_mean = session.run(pi.mean, {pi.obs: paths["obs"]})
        old_std = np.tile(session.run(pi.std), (len(paths["rwd"]), 1))
        for epoch in range(epochs):
            for batch_idx in minibatch_idx_list(batch_size, len(paths["rwd"])):
                dct = {obs: paths["obs"][batch_idx],
                       nobs: paths["nobs"][batch_idx],
                       act: paths["act"][batch_idx],
                       rwd: paths["rwd"][batch_idx],
                       done: paths["done"][batch_idx],
                       pi.act: paths["act"][batch_idx],
                       alpha: alpha_value}
                session.run([optimize_v, optimize_q], dct)
                session.run(update_vt)

        for epoch in range(epochs):
            for batch_idx in minibatch_idx_list(batch_size, len(paths["rwd"])):
                dct = {obs: paths["obs"][batch_idx],
                       act: paths["act"][batch_idx],
                       pi.act: paths["act"][batch_idx]}
                session.run([optimize_pi], dct)

        # Evaluate pi
        # layers_m = session.run(mean.vars)
        # draw_fast = lambda x : fast_policy(x, layers_m, act_bound=act_bound)
        # avg_rwd = evaluate_policy(env, policy=draw_fast, min_paths=paths_eval)
        avg_rwd = np.sum(paths["rwd"]) / paths["nb_paths"]
        entr = pi.estimate_entropy(paths["obs"])
        kl = pi.estimate_klm(paths["obs"], old_mean, old_std)
        mstde = session.run(loss_q, {obs: paths["obs"], nobs: paths["nobs"], act: paths["act"], rwd: paths["rwd"], done: paths["done"]})
        print('%d | %e   %e   %e   %e   ' % (itr, entr, kl, avg_rwd, mstde), flush=True)
        with open(logger.fullname, 'ab') as f:
            np.savetxt(f, np.atleast_2d([entr, kl, avg_rwd, mstde])) # save data

    session.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
