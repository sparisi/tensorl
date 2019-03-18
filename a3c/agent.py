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
import tensorflow_probability as tfp
import numpy as np

from common.data_collection import *

def update_target_graph(from_scope, to_scope):
	'''
	Copies one set of variables to another.
	Used to set worker network parameters to those of the master network.
	'''
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

	op = []
	for from_var, to_var in zip(from_vars, to_vars):
		op.append(to_var.assign(from_var))
	return op


class AC_Network():
	def __init__(self, scope, obs_size, act_size, act_bound=np.inf, optimizer=None, precision=tf.float32):
		with tf.variable_scope(scope):
			with tf.variable_scope('pi'):
				self.obs = tf.placeholder(shape=[None,obs_size], dtype=precision)
				last_out = self.obs
				last_out = tf.layers.dense(last_out, 15, activation=tf.nn.tanh)
				last_out = tf.layers.dense(last_out, 45, activation=tf.nn.tanh)
				last_out = tf.layers.dense(last_out, act_size, activation=None, kernel_initializer=tf.initializers.random_normal(0.0, 1e-8))
				self.mean = last_out
				if not np.any(np.isinf(act_bound)):
					self.mean = act_bound*tf.nn.tanh(self.mean)
				self.std = tf.Variable(4.*tf.ones([1,act_size], dtype=precision), dtype=precision)
				act_dist = tfp.distributions.MultivariateNormalDiag(self.mean, self.std)
				self.pi = act_dist.sample()
				self.act = tf.placeholder(dtype=precision, shape=[None,act_size])

			with tf.variable_scope('v'):
				last_out = self.obs
				last_out = tf.layers.dense(last_out, 15, activation=tf.nn.tanh)
				last_out = tf.layers.dense(last_out, 45, activation=tf.nn.tanh)
				last_out = tf.layers.dense(last_out, 1, activation=None, kernel_initializer=tf.initializers.random_normal(0.0, 1e-8))
				self.v = last_out

			# Only workers need ops for loss functions and gradient updates
			if scope != 'master':
				# Local losses
				self.target_v = tf.placeholder(shape=[None,1], dtype=precision)
				self.advantage = tf.placeholder(shape=[None,1], dtype=precision)
				self.loss_v = tf.losses.mean_squared_error(self.target_v, self.v)
				log_prob = tf.expand_dims(act_dist.log_prob(self.act), axis=-1)
				self.loss_pi = -tf.reduce_mean(tf.exp(log_prob)*self.advantage)

				# Apply worker losses to master nets
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')

				gradients = tf.gradients(self.loss_v + self.loss_pi, local_vars) # V and pi do not share variables, so we can sum them
				self.apply_grads = optimizer.apply_gradients(zip(gradients, global_vars))



class Master():
	def __init__(self, env_name, global_steps):
		self.name = 'master'
		self.global_steps = global_steps
		self.env = gym.make(env_name)

		obs_size = self.env.observation_space.shape[0]
		act_size = self.env.action_space.shape[0]
		act_bound = np.asscalar(self.env.action_space.high[0])
		if act_bound != -np.asscalar(self.env.action_space.low[0]):
			act_bound = np.inf

		self.obs_size = obs_size
		self.act_size = act_size
		self.ac_nets = AC_Network(self.name, obs_size, act_size, act_bound=act_bound)

	def run(self, session, coord, max_steps=1e6, eval_every=1, paths_eval=20):
		# The master checks the timesteps limit and evaluates itself every X steps
		print("Starting " + self.name)
		with session.as_default(), session.graph.as_default():
			last_update_at = 0
			while True:
				global_steps = session.run(self.global_steps)
				if global_steps - last_update_at > eval_every:
					policy_det = lambda x: np.squeeze(session.run(self.ac_nets.mean, {self.ac_nets.obs: np.atleast_2d(x)}))
					avg_rwd = evaluate_policy(self.env, policy=policy_det, min_paths=paths_eval)
					print("%d | %e" % (global_steps, avg_rwd))
					local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
					last_update_at = global_steps

				if global_steps > max_steps:
					coord.request_stop()
					break



class Worker():
	def __init__(self, number, env_name, optimizer, global_steps):
		self.name = 'worker_' + str(number)
		self.optimizer = optimizer
		self.global_steps = global_steps
		self.env = gym.make(env_name)
		self.increment_global_steps = self.global_steps.assign_add(1)

		obs_size = self.env.observation_space.shape[0]
		act_size = self.env.action_space.shape[0]
		act_bound = np.asscalar(self.env.action_space.high[0])
		if act_bound != -np.asscalar(self.env.action_space.low[0]):
			act_bound = np.inf

		self.obs_size = obs_size
		self.act_size = act_size
		self.ac_nets = AC_Network(self.name, obs_size, act_size, act_bound=act_bound, optimizer=optimizer)
		self.update_local_ops = update_target_graph('master', self.name) # Op to copy master paramters to worker networks

	def train(self, data, v, session, gamma):
		# Advantage estimation
		adv = np.empty_like(data["rwd"])
		for k in reversed(range(len(data["rwd"]))):
			if data["done"][k]:
				adv[k] = data["rwd"][k] - v[k]
			else:
				adv[k] = data["rwd"][k] + gamma * v[k+1] - v[k]

		# Update the master network using gradients from worker loss
		feed_dict = {self.ac_nets.target_v: np.atleast_2d(adv + v[:-1]),
			self.ac_nets.obs: np.atleast_2d(data["obs"]),
			self.ac_nets.act: np.atleast_2d(data["act"]),
			self.ac_nets.advantage: np.atleast_2d(adv)}
		session.run(self.ac_nets.apply_grads, feed_dict)

	def run(self, session, coord, max_ep_steps=1000, gamma=0.99, update_freq=2):
		print("Starting " + self.name)
		data = {}
		data["obs"] = np.zeros((update_freq,self.obs_size))
		data["nobs"] = np.zeros((update_freq,self.obs_size))
		data["act"] = np.zeros((update_freq,self.act_size))
		data["rwd"] = np.zeros((update_freq,1))
		data["done"] = np.zeros((update_freq,1))

		with session.as_default(), session.graph.as_default():
			while not coord.should_stop():
				session.run(self.update_local_ops) # Sync target with master

				step = 0
				data_idx = 0
				done = False
				obs = self.env.reset()

				while not done and step < max_ep_steps: # Episode loop
					act = np.squeeze(session.run(self.ac_nets.pi, {self.ac_nets.obs: np.atleast_2d(obs)}))
					nobs, rwd, done, _ = self.env.step(np.clip(act, self.env.action_space.low, self.env.action_space.high))
					step += 1
					if step == max_ep_steps:
						done = True # Like gym, which returns True when time limit is reached
					data["obs"][data_idx,:] = obs
					data["act"][data_idx,:] = act
					data["nobs"][data_idx,:] = nobs
					data["rwd"][data_idx,:] = rwd
					data["done"][data_idx,:] = done
					data_idx += 1
					obs = nobs
					session.run(self.increment_global_steps)

					# Update when the data is full or when the episode ends
					if data_idx == update_freq or done:
						batch = {}
						for k, _ in data.items(): # Remove empty zeros
							batch[k] = data[k][:data_idx,:]
						v = session.run(self.ac_nets.v, {self.ac_nets.obs: np.atleast_2d(batch["obs"])})

						if ~done: # Bootstrap the last step value with V[s']
							v = np.append(v, session.run(self.ac_nets.v, {self.ac_nets.obs: np.atleast_2d(nobs)}), axis=0)
						else:
							v = np.append(v, 0., axis=0)

						self.train(batch, v, session, gamma)

						data_idx = 0 # Reset index
						for k, _ in data.items(): # Reset data
							data[k] = np.zeros((update_freq,data[k].shape[1]))

						session.run(self.update_local_ops) # Sync target with master
