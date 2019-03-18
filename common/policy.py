import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np



class RandPolicy:
    '''
    Random normal policy.
    '''
    def __init__(self, act_size, std, name='pi'):
        self.act_size = act_size
        self.std = std
        self.name = 'rand_policy_' + name

    def draw_action(self, obs):
        return np.squeeze(np.random.normal(loc=0.0, scale=self.std, size=(np.atleast_2d(obs).shape[0],self.act_size)))



class MVNPolicy:
    '''
    Gaussian policy with diagonal covariance. The mean and the std can be any
    kind of tensor (a MLP depending on the state, a simple tensor, or a fixed constant).
    '''
    def __init__(self, session, obs, mean, std, name='pi', act_bound=np.inf):
        self.session = session
        self.name = 'mvn_policy_' + name
        self.obs = obs
        # If the environment has bounded actions, bound the policy output as well
        if not np.any(np.isinf(act_bound)):
            self.mean = act_bound*tf.nn.tanh(mean)
        else:
            self.mean = mean
        self.act_size = mean.get_shape().as_list()[1]
        self.std = std
        self.act_bound = act_bound

        self.act_dist = tfp.distributions.MultivariateNormalDiag(self.mean, self.std)
        self.output = self.act_dist.sample()

        self.entropy = tf.reduce_mean(self.act_dist.entropy())
        # self.entropy = 0.5 * tf.reduce_mean(self.act_size * (np.log(2.*np.pi) + 1.) + 2.*tf.reduce_sum(tf.log(self.std), axis=1))

        self.act = tf.placeholder(dtype=obs.dtype, shape=[None, self.act_size], name=name+'_act')
        self.log_prob = tf.expand_dims(self.act_dist.log_prob(self.act), axis=-1) # expand vector returned by log_prob to row vector
        # self.log_prob = -0.5*tf.reduce_sum((self.mean - self.act)**2 / self.std**2 + 2.*tf.reduce_sum(tf.log(self.std)) + self.act_size*np.log(2.*np.pi), axis=1, keepdims=True)

        self.old_mean = tf.placeholder(dtype=obs.dtype, shape=[None, self.act_size], name=name+'_old_mean')
        self.old_std = tf.placeholder(dtype=obs.dtype, shape=[None, self.act_size], name=name+'_old_std')

        # KL divergences: m-projection KL(pi_new || pi_old) and i-projection KL(pi_old || pi_new)
        self.klm = 0.5 * tf.reduce_mean(2.*tf.reduce_sum(tf.log(self.std), axis=1) - 2.*tf.reduce_sum(tf.log(self.old_std), axis=1) +
                                tf.reduce_sum((self.mean - self.old_mean)**2 / self.std**2, axis=1) +
                                tf.reduce_sum(self.old_std**2/self.std**2, axis=1) - self.act_size)
        self.kli = 0.5 * tf.reduce_mean(2.*tf.reduce_sum(tf.log(self.old_std), axis=1) - 2.*tf.reduce_sum(tf.log(self.std), axis=1) +
                                tf.reduce_sum((self.old_mean - self.mean)**2 / self.old_std**2, axis=1) +
                                tf.reduce_sum(self.std**2/self.old_std**2, axis=1) - self.act_size)


    def get_log_prob(self, obs, act):
        return self.session.run(self.log_prob, {self.obs: np.atleast_2d(obs), self.act: np.atleast_2d(act)})

    def estimate_entropy(self, obs):
        return np.squeeze(self.session.run(self.entropy, {self.obs: np.atleast_2d(obs)}))

    def estimate_kli(self, obs, old_mean, old_std):
        return np.squeeze(self.session.run(self.kli, {self.obs: np.atleast_2d(obs), self.old_std: np.atleast_2d(old_std), self.old_mean: np.atleast_2d(old_mean)}))

    def estimate_klm(self, obs, old_mean, old_std):
        return np.squeeze(self.session.run(self.klm, {self.obs: np.atleast_2d(obs), self.old_std: np.atleast_2d(old_std), self.old_mean: np.atleast_2d(old_mean)}))

    def draw_action(self, obs):
        return np.squeeze(self.session.run(self.output, {self.obs: np.atleast_2d(obs)}))

    def draw_action_det(self, obs):
        return np.squeeze(self.session.run(self.mean, {self.obs: np.atleast_2d(obs)}))




class SoftmaxPolicy:
    '''
    pi(a_i|s) = exp(temp*f(s,a_i)) / sum(temp*f(s,a))
    '''
    def __init__(self, session, obs, f, n_act, log_temp=0., name='pi'):
        self.session = session
        self.name = 'softmax_policy_' + name
        self.obs = obs
        self.n_act = n_act
        self.f = f

        self.log_temp = tf.Variable(log_temp, trainable=False, name='pi_log_temp')
        self.max_temp = tf.Variable(tf.exp(log_temp), trainable=False, name='pi_max_temp')
        self.temp = tf.minimum(tf.exp(self.log_temp), self.max_temp)
        self.action_map = tf.Variable(tf.ones([self.n_act, self.n_act]) / self.n_act, name='pi_action_map') # normalization factor to ensure that the initial policy is uniform irrespective of f or the temperature
        self.act_logits = self.temp * tf.matmul(self.f, self.action_map)
        self.act_dist = tfp.distributions.Categorical(logits=self.act_logits)
        self.output = self.act_dist.sample()

        self.entropy = tf.reduce_mean(self.act_dist.entropy())

        self.act = tf.placeholder(dtype=obs.dtype, shape=[None, 1])
        self.log_prob = tf.expand_dims(self.act_dist.log_prob(tf.squeeze(self.act, axis=-1)), axis=-1) # expand vector returned by log_prob to row vector


    def draw_action(self, obs):
        return np.squeeze(self.session.run(self.output, {self.obs: np.atleast_2d(obs)}))

    def draw_action_det(self, obs):
        return np.argmax(self.session.run(self.f, {self.obs: np.atleast_2d(obs)}), axis=1)

    def get_log_prob(self, obs, act):
        return self.session.run(self.log_prob, {self.obs: np.atleast_2d(obs), self.act: np.atleast_2d(act)})

    def estimate_entropy(self, obs):
        return self.session.run(self.entropy, {self.obs: np.atleast_2d(obs)})





def fast_policy(obs, mean_layers, std_layers=None, act_bound=np.inf):
    '''
    Fast MVN "draw_action", useful for collecting actions or evaluating a policy
    when the mean (and, eventually, also the std) are approximated by neural networks.
    Instead of calling tf.session.run at each observation, we call it once to
    pre-compute the weights of the networks. The networks output (i.e., mean
    and std) are then computed just by matrix multiplication.

    Note that
    * it works only with single observations (no batches),
    * as for MVNPolicy, the covariance is diagonal,
    * the networks are assumed to have bias,
    * the activation function has to be set manually (currently, it is tanh),
    * the last activation is always assumed None (i.e., linear layer),
    * std_layers can be a list of layers (if a network is used), or just a matrix (if a variable is used).

    Example:

        draw = lambda x : np.squeeze(session.run(mean.output[0], {obs: np.atleast_2d(x)}))
        avg_rwd = evaluate_policy(env, draw, 1000)

        layers = session.run(mean.vars)
        draw_fast = lambda x : fast_policy(x, layers)
        avg_rwd = evaluate_policy(env, draw_fast, 1000)

    This trick makes sampling and evaluating 3-5 times faster.
    '''
    mean = obs
    std = obs
    for i in range(0,len(mean_layers)-2,2):
        mean = mean.dot(mean_layers[i])
        mean = mean + mean_layers[i+1][:,None].T
        mean = np.tanh(mean)
    mean = mean.dot(mean_layers[-2])
    mean = mean + mean_layers[-1][:,None].T

    if not np.any(np.isinf(act_bound)):
        mean = act_bound*np.tanh(mean)

    if std_layers is None:
        return mean.flatten()

    if not isinstance(std_layers,list):
        std = std_layers
    else:
        for i in range(0,len(std_layers)-2,2):
            std = std.dot(std_layers[i])
            std = std + std_layers[i+1][:,None].T
            std = np.tanh(std)
        std = std.dot(std_layers[-2])
        std = std + std_layers[-1][:,None].T

    return np.squeeze((mean + std*np.random.randn(std.shape[0],std.shape[1])).flatten())
