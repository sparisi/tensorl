'''
This script uses JOBLIB to run in parallel many trials of the same algorithm
on different environments with fixed random seed (seed = trial number).

Command
    python3 run_joblib.py <ALG_NAME> <N_JOBS> <N_TRIALS> <ENV_LIST>

Example
    python3 run_joblib.py 4 2 ddpg.ddpg Pendulum-v0 Swimmer-v2

Up to <N_JOBS> jobs will run in parallel. Pass -1 to use all CPUs and run as
many parallel trials as possible.

Data is still saved as usual in `data-trial`, but instead of the current date and time,
the seed (= trial number) is used. For example, for the above run data will be saved in

data-trial/ddpg/Pendulum-v0/0.dat
data-trial/ddpg/Pendulum-v0/1.dat
...

'''

import sys
from joblib import Parallel, delayed

alg_name = sys.argv[1]
n_jobs = int(sys.argv[2])
n_trials = int(sys.argv[3])
env_list = sys.argv[4:]

from importlib import import_module
alg = import_module(alg_name)

# create a list of arguments, one for each run
args = []
for trial in range(n_trials):
    for env_name in env_list:
        args.append((env_name, trial, trial))

# submit jobs
Parallel(n_jobs=n_jobs, verbose=0)((delayed(alg.main)(arg1,arg2,arg3) for (arg1,arg2,arg3) in args))
