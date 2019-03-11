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

from common import *

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
	def __init__(self, session, scope, obs_size, act_size, e_clip=0.1, act_bound=np.inf, optimizer_pi=None, optimizer_v=None, precision=tf.float32):
		self.obs = tf.placeholder(shape=[None,obs_size], dtype=precision)

		# Build pi
		self.mean = MLP([self.obs], [15, 45, act_size], [tf.nn.tanh, tf.nn.tanh, None], scope+'/pi')
		with tf.variable_scope(scope+'/pi'): std = tf.Variable(4.*tf.ones([1,act_size], dtype=precision), dtype=precision)
		self.pi = MVNPolicy(session, self.obs, self.mean.output[0], std, act_bound=act_bound)

		# Build V
		self.v = MLP([self.obs], [15, 45, 1], [tf.nn.tanh, tf.nn.tanh, None], scope+'/v')

		# Only workers need ops for loss functions and gradient updates
		if scope != 'master':
			# Local losses
			self.target_v = tf.placeholder(shape=[None,1], dtype=precision)
			self.loss_v = tf.losses.mean_squared_error(self.target_v, self.v.output[0])
			self.advantage = tf.placeholder(shape=[None,1], dtype=precision)
			self.old_log_probs = tf.placeholder(dtype=precision, shape=[None,1])
			prob_ratio = tf.exp(self.pi.log_prob - self.old_log_probs)
			clip_pr = tf.clip_by_value(prob_ratio, 1.-e_clip, 1.+e_clip)
			self.loss_pi = -tf.reduce_mean(tf.minimum(tf.multiply(prob_ratio, self.advantage), tf.multiply(clip_pr, self.advantage)))

			# Apply worker losses to master nets
			global_vars_pi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master/pi')
			global_vars_v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master/v')
			gradients_pi = tf.gradients(self.loss_pi, self.mean.vars+[std])
			gradients_v = tf.gradients(self.loss_v, self.v.vars)
			self.apply_grads_pi = optimizer_pi.apply_gradients(zip(gradients_pi, global_vars_pi))
			self.apply_grads_v = optimizer_v.apply_gradients(zip(gradients_v, global_vars_v))


class Master():
	def __init__(self, session, env_name, global_steps):
		self.name = 'master'
		self.session = session
		self.global_steps = global_steps
		self.env = gym.make(env_name)

		obs_size = self.env.observation_space.shape[0]
		act_size = self.env.action_space.shape[0]
		act_bound = np.asscalar(self.env.action_space.high[0])
		if act_bound != -np.asscalar(self.env.action_space.low[0]):
			act_bound = np.inf

		self.obs_size = obs_size
		self.act_size = act_size
		self.ac_nets = AC_Network(session, self.name, obs_size, act_size, act_bound=act_bound)

	def run(self, coord, max_steps=1e6, eval_every=10000, paths_eval=20):
		# The master checks the timesteps limit and evaluates itself every X steps
		print("Starting " + self.name)
		with self.session.as_default(), self.session.graph.as_default():
			last_update_at = 0
			while True:
				global_steps = self.session.run(self.global_steps)
				if global_steps - last_update_at > eval_every:
					avg_rwd = evaluate_policy(self.env, policy=self.ac_nets.pi.draw_action, min_paths=paths_eval)
					print("%d | %e" % (global_steps, avg_rwd))
					last_update_at = global_steps

				if global_steps > max_steps:
					coord.request_stop()
					break



class Worker():
	def __init__(self, session, number, env_name, optimizer_pi, optimizer_v, global_steps):
		self.name = 'worker_' + str(number)
		self.session = session
		self.optimizer_pi = optimizer_pi
		self.optimizer_v = optimizer_v
		self.global_steps = global_steps
		self.env = gym.make(env_name)
		self.increment = tf.placeholder(dtype=tf.int32)
		self.increment_global_steps = self.global_steps.assign_add(self.increment)

		obs_size = self.env.observation_space.shape[0]
		act_size = self.env.action_space.shape[0]
		act_bound = np.asscalar(self.env.action_space.high[0])
		if act_bound != -np.asscalar(self.env.action_space.low[0]):
			act_bound = np.inf

		self.obs_size = obs_size
		self.act_size = act_size
		self.ac_nets = AC_Network(session, self.name, obs_size, act_size, act_bound=act_bound, optimizer_pi=optimizer_pi, optimizer_v=optimizer_v)
		self.update_local_ops = update_target_graph('master', self.name) # op to copy master paramters to worker networks

	def run(self, coord, min_trans=400, gamma=0.99, lambda_trace=0.95, epochs=20, batch_size=64):
		print("Starting " + self.name)

		with self.session.as_default(), self.session.graph.as_default():
			while not coord.should_stop():
				self.session.run(self.update_local_ops) # sync target with master

				# Collect data
				paths = collect_samples(self.env, policy=self.ac_nets.pi.draw_action, min_trans=min_trans)
				nb_trans = len(paths["rwd"])
				self.session.run(self.increment_global_steps, {self.increment: nb_trans})

				# Update V
				for epoch in range(epochs):
					v_values = self.session.run(self.ac_nets.v.output[0], {self.ac_nets.obs: paths["obs"]})
					a_values = gae(paths, v_values, gamma, lambda_trace) # compute the advantage
					target_values = v_values + a_values # generalized Bellman operator
					for batch_idx in minibatch_idx_list(batch_size, nb_trans):
						dct = {self.ac_nets.target_v: target_values[batch_idx],
							self.ac_nets.obs: paths["obs"][batch_idx]}
						self.session.run(self.ac_nets.apply_grads_v, dct)

				v_values = self.session.run(self.ac_nets.v.output[0], {self.ac_nets.obs: paths["obs"]})
				a_values = gae(paths, v_values, gamma, lambda_trace) # compute the advantage
				a_values = (a_values - np.mean(a_values)) / np.std(a_values)

				# Update pi
				olp_values = self.session.run(self.ac_nets.pi.log_prob, {self.ac_nets.obs: paths["obs"], self.ac_nets.pi.act: paths["act"]})
				for epoch in range(epochs):
					for batch_idx in minibatch_idx_list(batch_size, nb_trans):
						dct = {self.ac_nets.obs: paths["obs"][batch_idx],
							self.ac_nets.pi.act: paths["act"][batch_idx],
							self.ac_nets.advantage: a_values[batch_idx],
							self.ac_nets.old_log_probs: olp_values[batch_idx]}
						self.session.run(self.ac_nets.apply_grads_pi, dct)
