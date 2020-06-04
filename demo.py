'''
Simple demo running an actor-critic-like REINFORCE (first fit Q, then run REINFORCE).

This demo shows:
* how to use the LQR environment from https://github.com/sparisi/gym_toy,
* how to use the plotting functions,
* how to save data,
* how to save/restore a model,
* how to save a summary to visualize the graph.

To see the graph, run

tensorboard --logdir logs-tf/

and go to http://127.0.1.1:6006/
'''

import gym, gym.spaces
import tensorflow as tf
import numpy as np
import sys
from common import *

dir_log_tf = './logs-tf/'
dir_model_tf = './models-tf/'

def main(checkpoint_path=None):
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

    # use vanilla normalized gradient
    optimizer   = tf.train.GradientDescentOptimizer(1e-2)
    gradients_unclip, variables = zip(*optimizer.compute_gradients(loss_pi, mean.vars+[std]))
    gradients_clip, _ = tf.clip_by_global_norm(gradients_unclip, 1.0)
    optimize_pi = optimizer.apply_gradients(zip(gradients_clip, variables))

    # init pi and Q to 0
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    q.reset(session, 0.)

    # prepare model saver
    saver = tf.train.Saver(max_to_keep=10)

    # restore all variables
    if checkpoint_path is not None:
        saver.restore(session, checkpoint_path)
        itr = int(checkpoint_path.split('-')[-1])
    else:
        itr = 1

    # save a log to visualize the graph with tensorboard
    tf.summary.FileWriterCache.clear()
    writer = tf.summary.FileWriter(dir_log_tf + env.env.__class__.__name__, graph=session.graph)
    writer.flush()
    writer.close()

    # prepare Q plot
    myplot = My3DPlot(-20,20,-20,20,"Q-function")

    # create file and folder to store data and model
    logger_data = LoggerData('reinforce', env.env.__class__.__name__, 'demo')
    logger_model = LoggerModel('reinforce', env.env.__class__.__name__, 'demo')


    # learning
    while itr < 100:
        paths = collect_samples(env, policy=pi.draw_action, min_trans=5000) # collect samples
        R = mc_ret(paths,gamma) # estimate returns

        optimizer_q.minimize(session, {obs: paths["obs"], act: paths["act"], targets: R}) # train Q
        session.run(optimize_pi, {obs: paths["obs"], pi.act: paths["act"], act: paths["act"]}) # train pi

        avg_rwd = env.env.avg_return(session.run(mean.vars)[0], gamma) # evaluate pi (LQR has a closed form solution if the policy is linear)
        myplot.update(session.run(q.output[0], {obs: np.atleast_2d(myplot.XY[:,0]).T, act: np.atleast_2d(myplot.XY[:,1]).T})) # show Q
        print(itr, avg_rwd, flush=True)

        with open(logger_data.fullname, 'ab') as f:
            np.savetxt(f, np.atleast_2d(avg_rwd)) # save data
        saver.save(session, logger_model.fullname, global_step=itr) # save all variables

        itr += 1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
