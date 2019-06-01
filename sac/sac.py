'''
SAC in DDPG style (the policy is updated after each step using mini-batches from a replay memory).
The critic is as in the first paper https://arxiv.org/pdf/1801.01290.pdf, i.e., there is a single
Q-function critic, which uses a target V-function critic for its update.
The entropy regularizer alpha is learned, as in the second paper https://arxiv.org/pdf/1812.05905.pdf.
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
    nobs = tf.placeholder(dtype=precision, shape=[None, obs_size], name='nobs')
    act = tf.placeholder(dtype=precision, shape=[None, act_size], name='act')
    rwd = tf.placeholder(dtype=precision, shape=[None, 1], name='rwd')
    done = tf.placeholder(dtype=precision, shape=[None, 1], name='done')

    # Build pi
    act_bound = np.asscalar(env.action_space.high[0])
    assert act_bound == -np.asscalar(env.action_space.low[0])
    mean = MLP([obs], pi_sizes+[act_size], pi_activations+[None], 'pi_mean')
    with tf.variable_scope('pi_std'): std = tf.Variable(std_noise * tf.ones([1, act_size], dtype=precision), dtype=precision)
    pi = MVNPolicy(session, obs, mean.output[0], std, act_bound=act_bound)

    # Build Q and V
    q = MLP([tf.concat([obs, act], axis=1),  # for TD error
             tf.concat([obs, pi.act], axis=1)], # for pi loss (no reparameterization)
             q_sizes+[1], q_activations+[None], 'q')
    v = MLP([obs], v_sizes+[1], v_activations+[None], 'v')
    vt = MLP([nobs], v_sizes+[1], v_activations+[None], 'target_v')

    # alpha optimization
    log_alpha = tf.Variable(0., dtype=precision)
    alpha = tf.exp(log_alpha)
    target_entropy = - act_size
    loss_alpha = - tf.reduce_mean(log_alpha * (pi.log_prob + target_entropy))
    optimize_alpha = tf.train.AdamOptimizer(lrate_alpha).minimize(loss_alpha, var_list=log_alpha)

    # V and Q optimization
    loss_v = tf.losses.mean_squared_error(v.output[0], q.output[1] - alpha * pi.log_prob)
    loss_q = tf.losses.mean_squared_error(q.output[0], rwd + gamma * vt.output[0] * (1. - done))
    optimize_v = tf.train.AdamOptimizer(lrate_v).minimize(loss_v, var_list=v.vars)
    optimize_q = tf.train.AdamOptimizer(lrate_q).minimize(loss_q, var_list=q.vars)

    # pi optimization
    loss_pi = tf.reduce_mean(pi.log_prob * tf.stop_gradient (alpha * pi.log_prob - (q.output[1] - v.output[0]))) # no reparameterization, V as baseline
    optimize_pi = tf.train.AdamOptimizer(lrate_pi).minimize(loss_pi, var_list=mean.vars+[std])

    # Init variables
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    v.reset(session, 0.)
    q.reset(session, 0.)

    # Update target network op
    update_vt = []
    for vars_v, vars_vt in zip(v.vars, vt.vars):
        session.run(tf.assign(vars_vt, vars_v))
        update_vt.append(tf.assign(vars_vt, tau_critic*vars_v + (1.-tau_critic)*vars_vt)) # soft target update

    # Init dataset
    paths = {}
    paths["obs"] = np.empty((int(max_trans),obs_size))
    paths["nobs"] = np.empty((int(max_trans),obs_size))
    paths["act"] = np.empty((int(max_trans),act_size))
    paths["rwd"] = np.empty((int(max_trans),1))
    paths["done"] = np.empty((int(max_trans),1))
    trans = 0
    data_idx = 0

    logger = LoggerData('sac', env_name, run_name)
    while trans < min_trans + learn_trans:
        # Reset environment
        obs_i = env.reset()
        done_i = False

        # Run episode
        while not done_i:
            act_i = pi.draw_action(obs_i)
            nobs_i, rwd_i, done_i, _ = env.step(np.minimum(np.maximum(act_i, env.action_space.low), env.action_space.high))

            paths["obs"][data_idx,:] = obs_i
            paths["nobs"][data_idx,:] = nobs_i
            paths["rwd"][data_idx,:] = rwd_i
            paths["act"][data_idx,:] = act_i
            paths["done"][data_idx,:] = done_i

            obs_i = nobs_i
            data_idx += 1
            trans += 1

            # Reset index to store data
            if data_idx >= max_trans:
                data_idx = 0

            # Gradient step if warmup is over
            if trans > min_trans:
                batch_idx = minibatch_idx(batch_size, min(len(paths["rwd"]), trans))
                dct = {obs: paths["obs"][batch_idx,:],
                       nobs: paths["nobs"][batch_idx,:],
                       act: paths["act"][batch_idx,:],
                       pi.act: session.run(pi.output, {obs: paths["obs"][batch_idx,:]}),
                       rwd: paths["rwd"][batch_idx],
                       done: paths["done"][batch_idx]}

                session.run(optimize_q, dct)
                session.run(optimize_v, dct)
                session.run(optimize_pi, dct)
                session.run(optimize_alpha, dct)
                session.run(update_vt)

            # Print info every X transitions (use the greedy policy, no noise)
            if trans % eval_every == 0 and trans > min_trans:
                avg_rwd = evaluate_policy(env, pi.draw_action_det, paths_eval, render=False)
                mstde = session.run(loss_q, {obs: paths["obs"], nobs: paths["nobs"], act: paths["act"], rwd: paths["rwd"], done: paths["done"]})
                entr = pi.estimate_entropy(paths["obs"])
                alpha_value = session.run(alpha)
                print('%d   %.4f   %.4f   %.4f   %.4f   ' % (trans, avg_rwd, mstde, entr, alpha_value), flush=True)
                with open(logger.fullname, 'ab') as f:
                    np.savetxt(f, np.atleast_2d([avg_rwd, mstde, entr, alpha_value])) # save data

    session.close()






if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
