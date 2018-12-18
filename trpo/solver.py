'''
https://github.com/joschu/modular_rl
https://github.com/MahanFathi/TRPO-TensorFlow
'''

import tensorflow as tf
import numpy as np


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)


class TRPO:
    def __init__(self, session, adv, pi, loss_pi, vars_pi, old_log_prob, kl_bound=1e-2, cg_damping=1e-1):
        self.session = session
        self.adv = adv
        self.pi = pi
        self.old_log_prob = old_log_prob
        self.params = vars_pi
        self.loss = loss_pi
        self.kl_bound = kl_bound
        self.cg_damping = cg_damping
        self.pg = flatgrad(self.loss, self.params)

        # First, get the tensor for the gradient-vector-product (gvp)
        # Then, get its derivative, that is the hessian-vector-product (hvp)
        self.shapes = [v.shape.as_list() for v in self.params]
        self.size_params = np.sum([np.prod(shape) for shape in self.shapes])
        self.p = tf.placeholder(self.adv.dtype, (self.size_params,)) # the vector
        grads = tf.gradients(self.pi.kl, self.params)
        tangents = []
        start = 0
        for shape in self.shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(self.p[start:start + size], shape))
            start += size
        gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)])
        self.hvp = flatgrad(gvp, self.params)

        # Update operations (reshape flat params and assign new value)
        self.flat_params = tf.concat([tf.reshape(param, [-1]) for param in self.params], axis=0)
        self.flat_params_place = tf.placeholder(self.adv.dtype, (self.size_params,))
        self.assign_weights_ops = []
        start = 0
        assert len(self.params) == len(self.shapes), "Wrong shapes."
        for i, shape in enumerate(self.shapes):
            size = np.prod(shape)
            param = tf.reshape(self.flat_params_place[start:start + size], shape)
            self.assign_weights_ops.append(self.params[i].assign(param))
            start += size
        assert start == self.size_params, "Wrong shapes."


    def assign_vars(self, params):
        self.session.run(self.assign_weights_ops, {self.flat_params_place: params})

    def get_flat_params(self):
        return self.session.run(self.flat_params)


    def step(self, obs, act, adv, old_log_prob, old_mean, old_std):
        dct = {self.pi.obs: obs,
                self.pi.act: act,
                self.adv: adv,
                self.pi.old_mean: old_mean,
                self.pi.old_std: old_std,
                self.old_log_prob: old_log_prob}

        prev_params = self.get_flat_params()

        def get_pg():
            return self.session.run(self.pg, dct)

        def get_hvp(p):
            dct[self.p] = p
            return self.session.run(self.hvp, dct) + self.cg_damping * p

        def get_loss(params):
            self.assign_vars(params)
            return self.session.run([self.loss, self.pi.kl], dct)

        pg = get_pg() # vanilla gradient
        if np.allclose(pg, 0):
            print("Got zero gradient. Not updating.")
            return
        stepdir = cg(get_vp=get_hvp, b=-pg) # natural gradient direction
        shs = 0.5 * stepdir.dot(get_hvp(stepdir))
        lm = np.sqrt(shs / self.kl_bound) # optimal stepsize (see Eq 3-5 in https://arxiv.org/pdf/1703.02660.pdf)
        fullstep = stepdir / lm
        expected_improve = -pg.dot(stepdir) / lm
        success, new_params = linesearch(get_loss, prev_params, fullstep, expected_improve, self.kl_bound)
        self.assign_vars(new_params)


def linesearch(f, x, fullstep, expected_improve_rate, kl_bound, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)[0]
    for stepfrac in (.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval, newkl = f(xnew)
        # if newkl > kl_bound:
        #     newfval += np.inf
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew
    return False, x


def cg(get_vp, b, cg_iters=10, residual_tol=1e-10):
    """
    Conjugate gradient method, approximately solves get_vp(x) = b for x
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = get_vp(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x
