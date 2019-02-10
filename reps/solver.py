import tensorflow as tf
import numpy as np

class REPS:

    def __init__(self, session, epsilon, v, obs, nobs, iobs, rwd, scipy_iter=100, l2reg=0.0, verbose=False):
        self.verbose = verbose
        self.session = session
        self.epsilon = epsilon
        self.v = v
        self.eta = tf.Variable(1e2, dtype=obs.dtype, name='reps_eta')
        self.theta = v.vars
        self.gamma = tf.placeholder(dtype=obs.dtype, name='gamma')
        self.obs = obs
        self.nobs = nobs
        self.iobs = iobs
        self.rwd = rwd
        self.adv = self.rwd + self.gamma*self.v.output[1] + (1.-self.gamma)*tf.reduce_mean(self.v.output[2]) - self.v.output[0]
        self.w = tf.exp((self.adv - tf.reduce_max(self.adv)) / self.eta)
        self.dual = self.eta * self.epsilon + self.eta * tf.log(tf.reduce_mean(self.w)) + tf.reduce_max(self.adv) + tf.reduce_mean([tf.nn.l2_loss((x)) for x in self.v.vars])*l2reg
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.dual,
                                              options={'maxiter': scipy_iter, 'disp': False, 'ftol': 0},
                                              method='SLSQP',
                                              var_list=self.theta+[self.eta],
                                              var_to_bounds={self.eta: (1e-8, np.infty)})
        self.reset_eta = tf.assign(self.eta,1e2)


    def optimize(self, obs, nobs, iobs, rwd, gamma):
        dct = {self.obs: obs, self.nobs: nobs, self.rwd: rwd, self.iobs: iobs, self.gamma: gamma}
        nb_trans = rwd.shape[0]

        self.session.run(self.reset_eta)

        # Print info
        if self.verbose:
            print()
            phi = np.array(self.session.run(self.v.phi[0], dct))
            nphi = np.array(self.session.run(self.v.phi[1], dct))
            iphi = np.array(self.session.run(self.v.phi[2], dct))

            dual = self.session.run(self.dual, dct)
            eta = self.session.run(self.eta)
            msadv = np.mean(self.session.run(self.adv, dct)**2) # mean squared advantage

            mu_diff = gamma*nobs + (1.-gamma)*np.mean(iobs,axis=0) - obs
            mu_std = np.std(obs,axis=0)
            mu_std[mu_std == 0] = 1.
            mu_err = np.max(np.abs(mu_diff/mu_std))

            phi_diff = gamma*nphi + (1.-gamma)*np.mean(iphi,axis=0) - phi
            phi_std = np.std(phi,axis=0)
            phi_std[phi_std == 0] = 1.
            phi_err = np.max(np.abs(phi_diff/phi_std))

            print('     MU ERR        PHI ERR       ETA           DUAL           MSA ')
            print('     %e  %e  %e  %e  %e ' % (mu_err, phi_err, eta, dual, msadv))

        self.optimizer.minimize(self.session, dct)

        # Compute weights and KL
        w = np.squeeze(self.session.run(self.w, dct))
        wsum = np.sum(w)
        w = w / wsum
        kl = np.nansum(w[np.nonzero(w)] * np.log(w[np.nonzero(w)] * w.size))

        # Print info
        if self.verbose:
            dual = self.session.run(self.dual, dct)
            eta = self.session.run(self.eta)
            msadv = np.mean(self.session.run(self.adv, dct)**2)

            mu_err = np.max(np.abs(np.dot(w,mu_diff)/mu_std))
            phi_err = np.max(np.abs(np.dot(w,phi_diff)/phi_std))

            print('     %e  %e  %e  %e  %e ' % (mu_err, phi_err, eta, dual, msadv))
            print()

        return kl, w*wsum
