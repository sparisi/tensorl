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
    max_ep_steps = 1000
    update_freq = 1
    # update_freq = max_ep_steps
    max_steps = 1e6
    lrate = 1e-4
    n_proc = multiprocessing.cpu_count()
    # n_proc = 3

    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)
        optimizer = tf.train.AdamOptimizer(lrate)
        master = Master(env_name, global_steps)
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
