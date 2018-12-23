'''
DDPG without policy target network (there is still a Q-target network).

   TDerr = Q(s,a;w) - (r + g*Q(s',pi(s';t);w'))
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
    pi = MLP([obs, nobs], pi_sizes+[act_size], pi_activations+[None], 'pi') # pi(s)


    # Bound actions
    act_bound = np.asscalar(env.action_space.high[0])
    assert act_bound == -np.asscalar(env.action_space.low[0])
    pi.output[0] = act_bound*tf.nn.tanh(pi.output[0])
    pi.output[1] = act_bound*tf.nn.tanh(pi.output[1])

    # Build Q
    q = MLP([tf.concat([obs, act], axis=1), # Q(s,a) to minimize the TD error
             tf.concat([obs, pi.output[0]], axis=1)], # Q(s,pi(s)) to maximize the avg return
             q_sizes+[1], q_activations+[None], 'q')

    qt = MLP([tf.concat([nobs, pi.output[1]], axis=1)], # Q(s',pi(s')) for the TD error targets
              q_sizes+[1], q_activations+[None], 'qt')

    # Loss functions, gradients and optimizers
    loss_q = tf.reduce_mean(0.5*tf.square( q.output[0] - (rwd + gamma * qt.output[0]) ))
    loss_pi = -tf.reduce_mean(q.output[1])


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





    # Init dataset
    paths = {}
    paths["obs"] = np.empty((int(max_trans),obs_size))
    paths["nobs"] = np.empty((int(max_trans),obs_size))
    paths["act"] = np.empty((int(max_trans),act_size))
    paths["rwd"] = np.empty((int(max_trans),1))
    paths["done"] = np.empty((int(max_trans),1))
    trans = 0
    data_idx = 0
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(act_size), sigma=float(std_noise)*np.ones(act_size))

    logger = LoggerData('ddpg_notarget', env_name, run_name)
    while trans < min_trans + learn_trans:
        # Reset environment
        obs_i = env.reset()
        done_i = False
        action_noise.reset()

        # Run episode
        while not done_i:
            act_i = np.squeeze(session.run(pi.output[0], {obs: np.atleast_2d(obs_i)})) + action_noise()
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
                       act: paths["act"][batch_idx,:],
                       nobs: paths["nobs"][batch_idx,:],
                       rwd: paths["rwd"][batch_idx],
                       done: paths["done"][batch_idx]}


                session.run(optimizer_q, dct)
                session.run(optimizer_pi, dct)
                session.run(update_qt)


            # Print info every X transitions (use the greedy policy, no noise)
            if trans % eval_every == 0 and trans > min_trans:
                layers = session.run(pi.vars)
                pi_det = lambda x : fast_policy(x, layers)
                avg_rwd = evaluate_policy(env, pi_det, paths_eval, render=False)
                td = session.run(loss_q, {obs: paths["obs"], act: paths["act"], rwd: paths["rwd"], done: paths["done"], nobs: paths["nobs"]})
                print('%d   %.4f   %.4f' % (trans, avg_rwd, td), flush=True)
                with open(logger.fullname, 'ab') as f:
                    np.savetxt(f, np.atleast_2d([avg_rwd, td])) # save data

    session.close()






if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
