'''
Trust region policy optimization https://arxiv.org/abs/1502.05477
'''

try:
    import roboschool
except ImportError:
    pass
try:
    import pybullet_envs
except ImportError:
    pass
import gym
import tensorflow as tf
import numpy as np
import sys

from common import *
from .hyperparameters import *
from .solver import *

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

    # Build pi
    act_bound = np.asscalar(env.action_space.high[0])
    assert act_bound == -np.asscalar(env.action_space.low[0])
    mean = MLP([obs], pi_sizes+[act_size], pi_activations+[None], 'pi_mean')
    with tf.variable_scope('pi_std'): std = tf.Variable(std_noise * tf.ones([1, act_size], dtype=precision), dtype=precision)
    pi = MVNPolicy(session, obs, mean.output[0], std, act_bound=act_bound)

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
    loss_pi = -tf.reduce_mean(tf.multiply(prob_ratio, advantage))
    solver = TRPO(session, advantage, pi, loss_pi, mean.vars+[std], old_log_probs, kl_bound=kl_bound, cg_damping=cg_damping)

    # Init variables
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    v.reset(session, 0.)

    logger = LoggerData('trpo', env_name, run_name)
    print()
    print('    V LOSS                         PI LOSS                        ENTROPY        KL             RETURN          MSTDE')
    for itr in range(maxiter):
        paths = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter)

        # Update V
        for epoch in range(epochs_v):
            v_values = session.run(v.output[0], {obs: paths["obs"]})
            a_values = gae(paths, v_values, gamma, lambda_trace) # compute the advantage
            target_values = v_values + a_values # generalized Bellman operator
            if epoch == 0:
                v_loss_before = session.run(loss_v, {obs: paths["obs"], target_v: target_values})
            for batch_idx in minibatch_idx_list(batch_size, len(target_values)):
                session.run(optimize_v, {obs: paths["obs"][batch_idx], target_v: target_values[batch_idx]})
        v_loss_after = session.run(loss_v, {obs: paths["obs"], target_v: target_values})

        # Estimate advantages
        v_values = session.run(v.output[0], {obs: paths["obs"]})
        a_values = gae(paths, v_values, gamma, lambda_trace)
        td_values = gae(paths, v_values, gamma, 0)
        mstde = np.mean(td_values**2)

        a_values = (a_values - np.mean(a_values)) / np.std(a_values)

        # Udpate pi
        old_lp = pi.get_log_prob(paths["obs"], paths["act"])
        old_mean = session.run(pi.mean, {pi.obs: paths["obs"]})
        old_std = np.tile(session.run(pi.std), (len(paths["rwd"]), 1))
        pi_loss_before = session.run(loss_pi, {obs: paths["obs"], pi.act: paths["act"], old_log_probs: old_lp, advantage: a_values})
        solver.step(paths["obs"], paths["act"], a_values, old_lp, old_mean, old_std)
        pi_loss_after = session.run(loss_pi, {obs: paths["obs"], pi.act: paths["act"], old_log_probs: old_lp, advantage: a_values})

        # Evaluate pi
        # layers_m = session.run(mean.vars)
        # draw_fast = lambda x : fast_policy(x, layers_m, act_bound=act_bound)
        # avg_rwd = evaluate_policy(env, policy=draw_fast, min_paths=paths_eval)
        avg_rwd = np.sum(paths["rwd"]) / paths["nb_paths"]
        entr = pi.estimate_entropy(paths["obs"])
        kl = pi.estimate_kl(paths["obs"], old_mean, old_std)
        print('%d | %e -> %e   %e -> %e   %e   %e   %e   %e   ' % (itr, v_loss_before, v_loss_after, pi_loss_before, pi_loss_after, entr, kl, avg_rwd, mstde), flush=True)
        with open(logger.fullname, 'ab') as f:
            np.savetxt(f, np.asmatrix([v_loss_before, v_loss_after, pi_loss_before, pi_loss_after, entr, kl, avg_rwd, mstde])) # save data

    session.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
