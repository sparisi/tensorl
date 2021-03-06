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
from .solver import *
from .hyperparameters import *

from sklearn.linear_model import Ridge

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
    env_eval = gym.make(env_name) # make a copy of env for evaluation

    # Init seeds
    seed = int(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)
    env_eval.seed(seed)

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth=True
    session = tf.Session(config=config_tf)

    # Init placeholders
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    obs = tf.placeholder(dtype=precision, shape=[None, obs_size], name='obs')
    q = tf.placeholder(dtype=precision, shape=[None, 1], name='q')

    # Compute Fourier features bandwidths as avg pairwise distance
    pi_rand = RandPolicy(act_size, std_noise, 'expl')
    paths_expl = collect_samples(env, policy=pi_rand.draw_action, min_trans=10000)

    from scipy.spatial.distance import pdist
    bw = []
    for i in range(obs_size):
        bw.append(np.mean(pdist(paths_expl["obs"][:,i][:,None])) + 1e-8)
    print()
    print('Fourier features bandwidths', bw)
    print()

    # Build pi
    weights = tf.placeholder(dtype=precision, shape=[None, 1], name='pi_weights') # for weighted max likelihood update
    mean = Fourier([obs], act_size, n_fourier, 'pi_mean', bandwidth=bw)
    with tf.variable_scope('pi_std'): std = tf.Variable(std_noise * tf.ones([1, act_size], dtype=precision), dtype=precision)
    pi = MVNPolicy(session, obs, mean.output[0], std) # with lin policy we don't bound the action, or we lose linearity and convexity
    loss_pi = -tf.reduce_mean(weights*pi.log_prob) #+ tf.reduce_mean([tf.nn.l2_loss((x)) for x in mean.vars+[std]])*0.00001 # weighted log-likelihood with l2 regularization

    # Define pi update ops
    new_mean_ph = tf.placeholder(dtype=precision, shape=mean.vars[0].get_shape().as_list(), name='new_mean')
    new_std_ph = tf.placeholder(dtype=precision, shape=[None, act_size], name='new_std')
    update_mean = tf.assign(mean.vars[0], new_mean_ph)
    update_std = tf.assign(std, new_std_ph)


    # Build V
    v = Fourier([obs], 1, n_fourier, 'v', bandwidth=bw)

    print("Number of policy parameters:", session.run(tf.size(mean.vars)))
    print("Number of V-function parameters:", session.run(tf.size(v.vars)))

    # Build REPS
    solver = ACREPS(session, epsilon, v, q, obs, verbose=verbose)

    # Init variables
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    v.reset(session, 0.)

    all_paths = []

    logger_data = LoggerData('acreps', env_name)
    for itr in range(maxiter):
        # Collect samples (at the beginning, collect as many paths as necessary according to max_reuse)
        if itr == 0:
            for r in range(max_reuse):
                paths_iter = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter)
                all_paths.append(paths_iter)
        else:
            paths_iter = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter)
            all_paths.append(paths_iter)

        if len(all_paths) > max_reuse:
            del all_paths[0]
        paths = merge_paths(all_paths)
        nb_trans = paths["rwd"].shape[0]

        # Run REPS
        kl, w = solver.optimize(paths["obs"], mc_ret(paths, gamma))

        old_mean = session.run(pi.mean, {pi.obs: paths["obs"]})
        old_std = session.run(pi.std)

        # ML update
        init_neg_lik = session.run(loss_pi, {pi.obs: np.atleast_2d(paths["obs"]), pi.act: np.atleast_2d(paths["act"]), weights: w[:,None]})

        # Weighted max lik policy update
        phi = session.run(mean.phi[0], {obs: paths["obs"]})
        clf = Ridge(alpha=1e-8, fit_intercept=False, solver='sparse_cg',
                    max_iter=2500, tol=1e-8)
        clf.fit(phi, paths["act"], sample_weight=w)
        new_K = clf.coef_
        Z = (np.square(np.sum(w, axis=0, keepdims=True)) -
             np.sum(np.square(w), axis=0, keepdims=True)) / \
            np.sum(w, axis=0, keepdims=True)
        tmp = paths["act"] - phi @ new_K.T
        new_cov = np.einsum('t,tk,th->kh', w, tmp, tmp) / (Z + 1e-8)
        session.run(update_mean, {new_mean_ph: new_K.T})
        session.run(update_std, {new_std_ph: np.sqrt(np.diag(new_cov))[None,:]*1.001})

        end_neg_lik = session.run(loss_pi, {pi.obs: np.atleast_2d(paths["obs"]), pi.act: np.atleast_2d(paths["act"]), weights: w[:,None]})
        actual_kl = pi.estimate_klm(paths["obs"], old_mean, old_std)

        # Evaluate pi and print info
        # avg_rwd = evaluate_policy(env_eval, policy=pi.draw_action_det, min_paths=1000)
        avg_rwd = np.sum(paths["rwd"]) / paths["nb_paths"]
        entr = pi.estimate_entropy(paths["obs"])
        print('%d | %.4f, %.4f, %.4f (%.4f), %.4f -> %.4f' % (itr, avg_rwd, entr, kl, actual_kl, init_neg_lik, end_neg_lik), flush=True)
        if verbose:
            print('--------------------------------------------------------------------------', flush=True)

        with open(logger_data.fullname, 'ab') as f:
            np.savetxt(f, np.atleast_2d([avg_rwd, entr, kl, actual_kl]))

    session.close()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(*sys.argv[1:])
    else:
        raise Exception('Missing environment!')
