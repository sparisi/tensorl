import tensorflow as tf
import numpy as np

class ACREPS:
    '''
    AC-REPS finds the temperature eta and the V-function maximizing E[A], with
    A = Q-V. Q is learned separately (Monte-Carlo estimates can also be used).
    The maximization has two constraint. The first bounds the KL divergence between
    the old and the new state-action distribution. The second enforces that the
    estimated state distribution does not change (this is done by matching feature averages).
    '''

    def __init__(self, session, epsilon, v, q, obs, scipy_iter=100, verbose=False):
        self.verbose = verbose
        self.session = session
        self.epsilon = epsilon
        self.eta = tf.Variable(1e0, dtype=obs.dtype, name='acreps_eta')
        self.obs = obs
        self.q = q
        self.v = v
        self.theta = v.vars
        self.adv = self.q - self.v.output[0]
        self.w = tf.exp((self.adv - tf.reduce_max(self.adv)) / self.eta)
        self.dual = self.eta*self.epsilon + self.eta*tf.log(tf.reduce_mean(self.w)) + tf.reduce_max(self.adv) + tf.reduce_mean(self.v.output[0])
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.dual,
                                              options={'maxiter': scipy_iter, 'disp': False, 'ftol': 0},
                                              method='SLSQP',
                                              var_list=self.theta+[self.eta],
                                              var_to_bounds={self.eta: (1e-8, np.infty)})

    def optimize(self, obs, q):
        dct = {self.obs: obs, self.q: q}
        nb_trans = q.shape[0]

        # Print info
        if self.verbose:
            print()
            print('   KL      ETA           DUAL           MSA           PHI ERR ')
            print('           %e  %e  %e ' % (self.session.run(self.eta), self.session.run(self.dual, dct), np.mean(self.session.run(self.adv, dct)**2)))

        self.optimizer.minimize(self.session, dct)

        # Compute weights and KL
        w = np.squeeze(self.session.run(self.w, dct))
        wsum = np.sum(w)
        w = w / wsum
        kl = np.nansum(w[np.nonzero(w)] * np.log(w[np.nonzero(w)] * w.size))

        if self.verbose:
            phi = np.array(self.session.run(self.v.phi[0], dct))
            phi_std = np.std(phi,axis=0)
            phi_std[phi_std == 0] = 1.
            phi_diff = np.dot(w,phi) - np.mean(phi,axis=0)
            phi_err = np.max(np.abs(phi_diff/phi_std))
            print('           %e  %e  %e  %e ' % (self.session.run(self.eta), self.session.run(self.dual, dct), np.mean(self.session.run(self.adv, dct)**2), phi_err))

            print()

        return kl, w*wsum
