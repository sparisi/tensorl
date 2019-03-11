'''
The code is written such that each worker gets a thread, and the master also gets
its own thread to evaluate its policy and to check when to stop the workers
(which happens by sending a Coordinator stop request).

The loss function is loss_pi + loss_v because V and pi do not share layers/parameters.
Master and workers have an AC_Network attribute where variables are managed
through scopes (master/v, worker_0/v, master/pi, ...).

Hyperparameters have to be set manually here (gamma, lrate, update_freq, ...)
and in 'agent.py' (networks architecture).

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

    gamma = 0.99
    update_freq = 2
    lrate = 1e-4
    max_steps = 1e6
    n_proc = multiprocessing.cpu_count()
    # n_proc = 3

    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)
        optimizer = tf.train.AdamOptimizer(lrate)
        master = Master(env_name, global_steps)
        max_ep_steps = master.env._max_episode_steps # Use default timesteps limit
        workers = []
        for i in range(n_proc-1):
            workers.append(Worker(i, env_name, optimizer, global_steps))

        coord = tf.train.Coordinator()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            threads = []
            master_work = lambda: master.run(session, coord, max_steps=max_steps)
            t = threading.Thread(target=(master_work))
            t.start()
            sleep(0.1)
            threads.append(t)

            for worker in workers:
                worker_work = lambda: worker.run(session, coord, max_ep_steps, gamma, update_freq=update_freq)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.1)
                threads.append(t)

            coord.join(threads)
