'''
This script uses MULTIPROCESSING to run in parallel many trials of the same algorithm
on different environments with fixed random seed (seed = trial number).

Command
    python3 run_multiproc.py <ALG_NAME> <N_TRIALS> <ENV_LIST>

Example
    python3 run_multiproc.py 5 ddpg.ddpg Pendulum-v0 Swimmer-v2

Data is still saved as usual in `data-trial`, but instead of the current date and time,
the seed (= trial number) is used. For example, for the above run data will be saved in

data-trial/ddpg/Pendulum-v0/0.dat
data-trial/ddpg/Pendulum-v0/1.dat
...

WARNING!
The script will run ALL trials in parallel! If you run too many
trials/environments this may clog your computer. Alternatively, you can use
`Pool` but there may be problems using it with Tensorflow.
Another option is to use `run_joblib.py`, but it is usually slower.
'''

import sys
from multiprocessing import Process

z = sys.argv[1]

from importlib import import_module
alg = import_module("reps.test")

# create a list of arguments, one for each run
args = []
for x in ["reps", "gae_scipy", "gae_adam"]:
    for y in ["reps", "td", "gae"]:
        # for z in ["reps_scipy", "reps_adam", "reps_scipy_b", "reps_adam_b", "trpo"]:
        args.append((x, y, z))

# submit procs
ps = []
for a in args:
    p = Process(target=alg.main, args=a)
    p.start()
    ps.append(p)

for p in ps:
    p.join()
