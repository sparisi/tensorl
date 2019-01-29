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

def main(seed=1, run_name=None):
    env_name = 'Pendulum-v0'

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
    env_eval._max_episode_steps = 150
    env._max_episode_steps = min_trans_per_iter # increase the time horizon
    env = make_average_env(env, reset_prob) # introduce state resets

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
    nobs = tf.placeholder(dtype=precision, shape=[None, obs_size], name='nobs')
    iobs = tf.placeholder(dtype=precision, shape=[None, obs_size], name='iobs')
    rwd = tf.placeholder(dtype=precision, shape=[None, 1], name='rwd')

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
    v = Fourier([obs, nobs, iobs], 1, n_fourier, 'v', bandwidth=bw)

    print("Number of policy parameters:", session.run(tf.size(mean.vars)))
    print("Number of V-function parameters:", session.run(tf.size(v.vars)))

    # Build REPS
    solver = REPS(session, epsilon, v, obs, nobs, iobs, rwd, verbose=verbose)

    # Init variables
    session.run(tf.global_variables_initializer())
    mean.reset(session, 0.)
    v.reset(session, 0.)

    all_paths = []
    gamma = 1. - reset_prob

    # Prepare plots
    myplot = My3DPlot(-np.pi,np.pi,-8,8,"V-function",100)
    myplot.ax_surf.change_geometry(1,3,1)
    myplot.ax_contour.change_geometry(1,3,2)
    ax_scatter = myplot.fig.add_subplot(1,3,3)
    XY = np.array([np.sin(myplot.XY[:,0]),np.cos(myplot.XY[:,0]),myplot.XY[:,1]]).T
    myplot_pi = My3DPlot(-np.pi,np.pi,-8,8,"PI",100)
    myplot_pi2 = My3DPlot(-np.pi,np.pi,-8,8,"PI CROP",100)

    logger_data = LoggerData('reps_pend', env_name)
    for itr in range(maxiter):
        # Collect samples (at the beginning, collect as many paths as necessary according to max_reuse)
        if itr == 0:
            for r in range(max_reuse):
                paths_iter = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter, max_trans_per_ep=max_trans)
                all_paths.append(paths_iter)
        else:
            paths_iter = collect_samples(env, policy=pi.draw_action, min_trans=min_trans_per_iter, max_trans_per_ep=max_trans)
            all_paths.append(paths_iter)

        if len(all_paths) > max_reuse:
            del all_paths[0]
        paths = merge_paths(all_paths)
        nb_trans = paths["rwd"].shape[0]

        avg_reset = paths_iter["nb_paths"] / np.sum(paths_iter["nb_steps"])
        if avg_reset < reset_prob*0.9:
            print('\n\033[93m Reset probabiliy too low (' + str(avg_reset) + ' / ' + str(reset_prob) + ').\033[0m')
        elif avg_reset > 1.1*reset_prob:
            print('\n\033[93m Reset probabiliy too high (' + str(avg_reset) + ' / ' + str(reset_prob) + ').\033[0m')

        # Run REPS
        kl, w = solver.optimize(paths["obs"], paths["nobs"], paths["iobs"], paths["rwd"]/np.std(paths["rwd"]), gamma)

        old_mean = session.run(pi.mean, {pi.obs: paths["obs"]})
        old_std = session.run(pi.std)
        init_neg_lik = session.run(loss_pi, {pi.obs: paths["obs"], pi.act: paths["act"], weights: w[:,None]})

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
        session.run(update_std, {new_std_ph: np.sqrt(np.diag(new_cov))[None,:]})

        end_neg_lik = session.run(loss_pi, {pi.obs: paths["obs"], pi.act: paths["act"], weights: w[:,None]})
        actual_kl = pi.estimate_kl(paths["obs"], old_mean, old_std)

        # Plot V, REPS weights, and pi
        ax_scatter.cla()
        angle = np.arctan2(paths["obs"][:,0], paths["obs"][:,1])
        ax_scatter.scatter(angle, paths["obs"][:,2], c=w)
        ax_scatter.set_title('REPS weights')
        myplot.update(session.run(v.output[0], {obs: np.atleast_2d(XY)}))
        myplot_pi.update(session.run(mean.output[0], {obs: np.atleast_2d(XY)}))
        myplot_pi2.update(np.clip(session.run(mean.output[0], {obs: np.atleast_2d(XY)}), -2, 2))

        # Fast deterministic policy
        bw = np.atleast_2d(bw)
        P = session.run(mean.P)
        shift = session.run(mean.shift)
        pi_det = lambda x: ( np.sin( (np.atleast_2d(x)/bw).dot(P) + shift) ).dot(new_K.T[1:,:]) + new_K.T[0,:]

        # Evaluate pi and print info
        avg_rwd = evaluate_policy(env_eval, policy=pi_det, min_paths=1000)
        # avg_rwd = np.sum(paths["rwd"]) / paths["nb_paths"]
        entr = pi.estimate_entropy(paths["obs"])
        print('%d | %.4f, %.4f, %.4f (%.4f), %.4f -> %.4f' % (itr, avg_rwd, entr, kl, actual_kl, init_neg_lik, end_neg_lik), flush=True)
        if verbose:
            print('--------------------------------------------------------------------------', flush=True)

        with open(logger_data.fullname, 'ab') as f:
            np.savetxt(f, np.atleast_2d([avg_rwd, entr, kl, actual_kl]))

    session.close()



if __name__ == '__main__':
    main(*sys.argv[1:])
