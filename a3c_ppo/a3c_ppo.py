'''
This implementation combines A3C and PPO.
Several agents (workers) collect samples on their own copy of the environment
with their own policy. Every worker iteration (= one iteration of PPO) the main
agent (master) gets V and pi updates using the workers gradients, and the workers
copy the master networks, as in A3C.

The code is written such that each worker gets a thread, and the master also gets
its own thread to evaluate its policy and to check when to stop the workers
(which happens by sending a Coordinator stop request).

Master and workers have an AC_Network attribute where variables are managed through scopes:
* The master's policy and V are under 'master/v' and 'master/pi',
* The workers' are under 'worker_0/v', 'worker_1/v', ...

Hyperparameters have to be set manually here (gamma, lambda_trace, lrate)
and in 'agent.py' (networks size/activation and e_clip).

This code was inspired by https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
'''

from .agent import *
import threading
import multiprocessing
from time import sleep
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
    else:
        raise Exception('Missing environment!')

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth=True
    session = tf.Session(config=config_tf)

    gamma = 0.99
    lambda_trace = 0.95
    lrate = 1e-3
    max_steps = 1e7
    n_proc = multiprocessing.cpu_count()
    # n_proc = 3

    with tf.device("/cpu:0"):
        global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)
        optimizer_pi = tf.train.AdamOptimizer(lrate)
        optimizer_v = tf.train.AdamOptimizer(lrate)
        master = Master(session, env_name, global_steps)
        workers = []
        for i in range(n_proc-1):
            workers.append(Worker(session, i, env_name, optimizer_pi, optimizer_v, global_steps))

        coord = tf.train.Coordinator()

        session.run(tf.global_variables_initializer())

        master.ac_nets.mean.reset(session, 0.)
        master.ac_nets.v.reset(session, 0.)
        for worker in workers:
            worker.ac_nets.mean.reset(session, 0.)
            worker.ac_nets.v.reset(session, 0.)

        threads = []
        master_work = lambda: master.run(coord, max_steps=max_steps)
        t = threading.Thread(target=(master_work))
        t.start()
        sleep(0.1)
        threads.append(t)

        for worker in workers:
            worker_work = lambda: worker.run(coord, gamma, lambda_trace)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.1)
            threads.append(t)

        coord.join(threads)
