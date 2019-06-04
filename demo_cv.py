'''
Simple demo running an actor-critic-like REINFORCE (first fit Q, then run REINFORCE).

This demo shows:
* how to use the LQR environment from https://github.com/sparisi/gym_toy,
* how to use the cross-validation function.
'''

import gym, gym.spaces
import tensorflow as tf
import numpy as np
from common import *

def main():
    session = tf.Session()

    # register and init LQR
    gym.envs.register(
         id='Lqr-v0',
         entry_point='gym.envs.gym_toy:LqrEnv',
         max_episode_steps=150,
         kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
    )
    env = gym.make('Lqr-v0')

    gamma = 0.99 # discount factor

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    obs      = tf.placeholder(dtype=tf.float32, shape=[None, obs_size], name='obs')
    act      = tf.placeholder(dtype=tf.float32, shape=[None, act_size], name='act')
    targets  = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='targets')

    # build Q
    q           = Quadratic([tf.concat([obs, act], axis=1)], 1, 'q')
    loss_q      = tf.reduce_mean(0.5*tf.square( q.output[0] - targets ))
    optimizer_q = tf.contrib.opt.ScipyOptimizerInterface(loss_q, options={'maxiter': 100, 'disp': False, 'ftol': 0}, method='SLSQP', var_list=q.vars)

    # build policy
    mean        = Linear([obs], act_size, 'pi_mean', use_bias=False)
    with tf.variable_scope('pi_std'): std = tf.Variable(10.0 * tf.ones([1, act_size]))
    pi          = MVNPolicy(session, obs, mean.output[0], std)
    loss_pi     = -tf.reduce_mean(tf.exp(pi.log_prob)*q.output[0])
    optimize_pi = tf.train.AdamOptimizer(1e-6).minimize(loss_pi)

    # init pi and Q to 0
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    q.reset(session, 0.)

    # create dummy variables for save/restore (cross-validation)
    last_vars = []
    for tmp in mean.vars+[std]:
        last_vars.append(tf.Variable(tmp))
    save_vars = []
    restore_vars = []
    for t, lt in zip(mean.vars+[std], last_vars):
        save_vars.append(tf.assign(lt, t))
        restore_vars.append(tf.assign(t, lt))

    # prepare plots to check test/train losses
    fig = plt.figure()
    fig.suptitle('Policy optimization')
    ax_train = fig.add_subplot(121)
    ax_test = fig.add_subplot(122)

    # learning
    itr = 1
    while itr < 100:
        paths = collect_samples(env, policy=pi.draw_action, min_trans=5000) # collect samples
        R = mc_ret(paths,gamma) # estimate returns

        optimizer_q.minimize(session, {obs: paths["obs"], act: paths["act"], targets: R}) # train Q

        dct_pi = {obs: paths["obs"], pi.act: paths["act"], act: paths["act"]}
        loss_test_history, loss_train_history = cross_validation(session, loss_pi, optimize_pi, save_vars, restore_vars, dct_pi) # train pi

        avg_rwd = env.env.avg_return(session.run(mean.vars)[0], gamma) # evaluate pi (LQR has a closed form solution if the policy is linear)

        print(itr, avg_rwd, flush=True) # print info

        ax_test.cla() # plot info
        ax_train.cla()
        ax_test.plot(np.arange(len(loss_test_history)),np.log(loss_test_history))
        ax_train.plot(np.arange(len(loss_train_history)),np.log(loss_train_history))
        ax_train.set_title('Train loss')
        ax_test.set_title('Test loss')
        plt.draw()
        plt.pause(0.0001)

        itr += 1


if __name__ == '__main__':
    main()
