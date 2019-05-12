'''
DDPG with prioritized experience replay.
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
    pi = MLP([obs], pi_sizes+[act_size], pi_activations+[None], 'pi') # pi(s)
    pit = MLP([nobs], pi_sizes+[act_size], pi_activations+[None], 'target_pi') # pi(s') used in the TD error targets

    # Bound actions
    act_bound = np.asscalar(env.action_space.high[0])
    assert act_bound == -np.asscalar(env.action_space.low[0])
    pi.output[0] = act_bound*tf.nn.tanh(pi.output[0])
    pit.output[0] = act_bound*tf.nn.tanh(pit.output[0])

    # Build Q
    q = MLP([tf.concat([obs, act], axis=1), # Q(s,a) to minimize the TD error
             tf.concat([obs, pi.output[0]], axis=1)], # Q(s,pi(s)) to maximize the avg return
             q_sizes+[1], q_activations+[None], 'q')

    qt = MLP([tf.concat([nobs, pit.output[0]], axis=1)], # Q(s',pi(s')) for the TD error targets
              q_sizes+[1], q_activations+[None], 'target_q')

    # Loss functions, gradients and optimizers
    td = q.output[0] - (rwd + gamma * qt.output[0] * (1.-done))
    weights = tf.placeholder(dtype=precision, shape=[None, 1], name='is_weights') # for importance sampling
    loss_q = tf.reduce_mean(weights * 0.5 * tf.square( q.output[0] - (rwd + gamma * qt.output[0] * (1.-done)) ))
    loss_pi = -tf.reduce_mean(weights * q.output[1])


    optimizer_q = tf.train.AdamOptimizer(lrate_q).minimize(loss_q, var_list=q.vars)
    optimizer_pi = tf.train.AdamOptimizer(lrate_pi).minimize(loss_pi, var_list=pi.vars)

    session.run(tf.global_variables_initializer())

    # Reset Q and pi to have almost-0 output
    q.reset(session, 0.)
    pi.reset(session, 0.)

    # Init target networks and prepare update operations
    update_qt = []
    for vars_q, vars_qt in zip(q.vars, qt.vars):
        session.run(tf.assign(vars_qt, vars_q))
        update_qt.append(tf.assign(vars_qt, tau_q*vars_q + (1.-tau_q)*vars_qt)) # soft target update
    update_pit = []
    for vars_pi, vars_pit in zip(pi.vars, pit.vars):
        session.run(tf.assign(vars_pit, vars_pi))
        update_pit.append(tf.assign(vars_pit, tau_pi*vars_pi + (1.-tau_pi)*vars_pit)) # soft target update

    # Init dataset
    paths = {}
    paths["obs"] = np.empty((int(max_trans),obs_size))
    paths["nobs"] = np.empty((int(max_trans),obs_size))
    paths["act"] = np.empty((int(max_trans),act_size))
    paths["rwd"] = np.empty((int(max_trans),1))
    paths["done"] = np.empty((int(max_trans),1))
    paths["prio"] = np.empty((int(max_trans),)) # priorities for PER
    trans = 0
    data_idx = 0
    action_noise = NormalActionNoise(mu=np.zeros(act_size), sigma=float(std_noise)*np.ones(act_size))

    logger = LoggerData('ddpg_per', env_name, run_name)
    while trans < min_trans + learn_trans:
        # Reset environment
        obs_i = env.reset()
        done_i = False

        # Run episode
        while not done_i:
            act_i = np.squeeze(session.run(pi.output[0], {obs: np.atleast_2d(obs_i)})) + action_noise()
            nobs_i, rwd_i, done_i, _ = env.step(np.minimum(np.maximum(act_i, env.action_space.low), env.action_space.high))

            paths["obs"][data_idx,:] = obs_i
            paths["nobs"][data_idx,:] = nobs_i
            paths["rwd"][data_idx,:] = rwd_i
            paths["act"][data_idx,:] = act_i
            paths["done"][data_idx,:] = done_i
            if trans < min_trans: # initial priorities are one or max of previous ones
                paths["prio"][data_idx] = 1.
            elif trans < max_trans:
                paths["prio"][data_idx] = np.max(paths["prio"][0:data_idx])
            else:
                paths["prio"][data_idx] = np.max(paths["prio"])

            obs_i = nobs_i
            data_idx += 1
            trans += 1

            # Reset index to store data
            if data_idx >= max_trans:
                data_idx = 0

            # Gradient step if warmup is over
            if trans > min_trans:
                alpha = 0.6
                beta_start = 0.4
                beta = min(1.0, beta_start + trans * (1.0 - beta_start) / (1. * learn_trans)) # anneal beta to 1 by the end of the learning
                N = min(len(paths["rwd"]), trans)

                probs = (paths["prio"][0:N] + 1e-5)**alpha # compute sampling probs
                probs = probs / np.sum(probs)
                w = (N * probs)**(-beta) # importance sampling weights
                w = w / np.max(w)
                batch_idx = minibatch_idx(batch_size, N, probs)
                dct = {obs: paths["obs"][batch_idx,:],
                       act: paths["act"][batch_idx,:],
                       nobs: paths["nobs"][batch_idx,:],
                       rwd: paths["rwd"][batch_idx],
                       done: paths["done"][batch_idx],
                       weights: w[batch_idx,None]}

                paths["prio"][batch_idx] = np.abs(session.run(td, dct).squeeze()) # use TD error for priorities

                session.run(optimizer_q, dct)
                session.run(optimizer_pi, dct)
                session.run(update_qt)
                session.run(update_pit)

            # Print info every X transitions (use the greedy policy, no noise)
            if trans % eval_every == 0 and trans > min_trans:
                layers = session.run(pi.vars)
                pi_det = lambda x : fast_policy(x, layers)
                avg_rwd = evaluate_policy(env, pi_det, paths_eval, render=False)
                mstde = session.run(loss_q, {obs: paths["obs"], act: paths["act"], rwd: paths["rwd"], done: paths["done"], nobs: paths["nobs"], weights: [[1.]]})
                print('%d   %.4f   %.4f' % (trans, avg_rwd, mstde), flush=True)
                with open(logger.fullname, 'ab') as f:
                    np.savetxt(f, np.atleast_2d([avg_rwd, mstde])) # save data

    session.close()






if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
