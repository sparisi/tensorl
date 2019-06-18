'''
Twin delayed DDPG. Differences from DDPG:
* There are two critics and the min of their Q is used in the TD targets
* In the TD targets, noise is added to pi(s';t')
* The policy is updated every 2 steps, not every step

   TDerr = Q(s,a;w_i) - (r + g*min_j(Q(s',pi(s';t')+noise;w_j')))
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
        print('\033[93m No hyperparameters defined for \"' + env_name + '\". Using default one.\033[0m')
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
    nact = tf.placeholder(dtype=precision, shape=[None, act_size], name='nact')
    rwd = tf.placeholder(dtype=precision, shape=[None, 1], name='rwd')
    done = tf.placeholder(dtype=precision, shape=[None, 1], name='done')


    # Build pi
    pi = MLP([obs], pi_sizes+[act_size], pi_activations+[None], 'pi') # pi(s)
    pit = MLP([nobs], pi_sizes+[act_size], pi_activations+[None], 'target_pi') # pi(s') used in the TD error targets

    # Bound actions
    act_bound = env.action_space.high
    assert np.all(act_bound == -env.action_space.low)
    pi.output[0] = act_bound*tf.nn.tanh(pi.output[0])
    pit.output[0] = act_bound*tf.nn.tanh(pit.output[0])

    # Build Q
    q1 = MLP([tf.concat([obs, act], axis=1), # Q1(s,a) to minimize the TD error
             tf.concat([obs, pi.output[0]], axis=1)], # Q1(s,pi(s)) to maximize the avg return
             q_sizes+[1], q_activations+[None], 'q1')
    q2 = MLP([tf.concat([obs, act], axis=1)], # only Q1 is used to update the policy
             q_sizes+[1], q_activations+[None], 'q2')

    q1t = MLP([tf.concat([nobs, nact], axis=1)], # Q1(s',a') for the TD error targets (a' = pi(s') + noise)
              q_sizes+[1], q_activations+[None], 'target_q1')
    q2t = MLP([tf.concat([nobs, nact], axis=1)],
              q_sizes+[1], q_activations+[None], 'target_q2')

    # Loss functions, gradients and optimizers
    loss_q1 = tf.reduce_mean(0.5*tf.square( q1.output[0] - (rwd + gamma * tf.minimum(q1t.output[0], q2t.output[0]) * (1.-done)) ))
    loss_q2 = tf.reduce_mean(0.5*tf.square( q2.output[0] - (rwd + gamma * tf.minimum(q1t.output[0], q2t.output[0]) * (1.-done)) ))
    loss_pi = -tf.reduce_mean(q1.output[1])

    optimizer_q1 = tf.train.AdamOptimizer(lrate_q).minimize(loss_q1, var_list=q1.vars)
    optimizer_q2 = tf.train.AdamOptimizer(lrate_q).minimize(loss_q2, var_list=q2.vars)
    optimizer_pi = tf.train.AdamOptimizer(lrate_pi).minimize(loss_pi, var_list=pi.vars)

    session.run(tf.global_variables_initializer())

    # Reset Q and pi to have almost-0 output
    # q1.reset(session, 0.) # having both Q to 0 may be detrimental
    # q2.reset(session, 0.)
    pi.reset(session, 0.)


    # TD3 hyperparameters as in its paper
    pi_delay = 2
    std_noise_n = 0.2 * act_bound
    nact_max = 0.5 * act_bound


    # Init target networks and prepare update operations
    update_qt = []
    for vars_q, vars_qt in zip(q1.vars, q1t.vars):
        session.run(tf.assign(vars_qt, vars_q))
        update_qt.append(tf.assign(vars_qt, tau_q*vars_q + (1.-tau_q)*vars_qt)) # soft target update
    for vars_q, vars_qt in zip(q2.vars, q2t.vars):
        session.run(tf.assign(vars_qt, vars_q))
        update_qt.append(tf.assign(vars_qt, tau_q*vars_q + (1.-tau_q)*vars_qt))
    update_pit = []
    for vars_pi, vars_pit in zip(pi.vars, pit.vars):
        session.run(tf.assign(vars_pit, vars_pi))
        update_pit.append(tf.assign(vars_pit, tau_pi*vars_pi + (1.-tau_pi)*vars_pit)) # soft target update

    # Init dataset
    paths = {}
    paths["obs"] = np.empty((int(max_trans),obs_size))
    paths["nobs"] = np.empty((int(max_trans),obs_size))
    paths["act"] = np.empty((int(max_trans),act_size))
    paths["nact"] = np.empty((int(max_trans),act_size))
    paths["rwd"] = np.empty((int(max_trans),1))
    paths["done"] = np.empty((int(max_trans),1))
    trans = 0
    data_idx = 0
    action_noise = NormalActionNoise(mu=np.zeros(act_size), sigma=float(std_noise)*np.ones(act_size))
    naction_noise = NormalActionNoise(mu=np.zeros(act_size), sigma=float(std_noise_n)*np.ones(act_size))

    logger = LoggerData('td3', env_name, run_name)
    while trans < min_trans + learn_trans:
        # Reset environment
        obs_i = env.reset()
        done_i = False

        # Run episode
        while not done_i:
            act_i = np.squeeze(session.run(pi.output[0], {obs: np.atleast_2d(obs_i)})) + action_noise()
            nobs_i, rwd_i, done_i, _ = env.step(np.minimum(np.maximum(act_i, env.action_space.low), env.action_space.high))
            nact_i = np.squeeze(session.run(pit.output[0], {nobs: np.atleast_2d(nobs_i)})) + np.clip(naction_noise(), -nact_max, nact_max)

            paths["obs"][data_idx,:] = obs_i
            paths["nobs"][data_idx,:] = nobs_i
            paths["rwd"][data_idx,:] = rwd_i
            paths["act"][data_idx,:] = act_i
            paths["nact"][data_idx,:] = nact_i
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
                       act: paths["act"][batch_idx,:],
                       nact: paths["nact"][batch_idx,:],
                       nobs: paths["nobs"][batch_idx,:],
                       rwd: paths["rwd"][batch_idx],
                       done: paths["done"][batch_idx]}


                session.run([optimizer_q1, optimizer_q2], dct)
                session.run(update_qt)
                if trans % pi_delay == 0:
                    session.run(optimizer_pi, dct)
                    session.run(update_pit)

            # Print info every X transitions (use the greedy policy, no noise)
            if trans % eval_every == 0 and trans > min_trans:
                layers = session.run(pi.vars)
                pi_det = lambda x : fast_policy(x, layers)
                avg_rwd = evaluate_policy(env, pi_det, paths_eval, render=False)
                mstde = session.run(loss_q1, {obs: paths["obs"], act: paths["act"], nact: paths["nact"], rwd: paths["rwd"], done: paths["done"], nobs: paths["nobs"]})
                print('%d   %.4f   %.4f' % (trans, avg_rwd, mstde), flush=True)
                with open(logger.fullname, 'ab') as f:
                    np.savetxt(f, np.atleast_2d([avg_rwd, mstde])) # save data

    session.close()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
