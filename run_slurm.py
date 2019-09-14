'''
This script uses SLURM to run in parallel many trials of the same algorithm
on different environments with fixed random seed (seed = trial number).

Command
    python3 run_slurm.py <ALG_NAME> <N_TRIALS> <ENV_LIST>

Example
    python3 run_slurm.py ddpg.ddpg 5 Pendulum-v0 Swimmer-v2

One job per run will be submitted.

Data is still saved as usual in `data-trial`, but instead of the current date and time,
the seed (= trial number) is used. For example, for the above run data will be saved in

data-trial/ddpg/Pendulum-v0/0.dat
data-trial/ddpg/Pendulum-v0/1.dat
...

Additionally, stdout and stderr are flushed to log files. For example

logs-trial/stdout_ddpg.ddpg_Pendulum-v0_0
logs-trial/stderr_ddpg.ddpg_Pendulum-v0_0
...

NOTE: Change the slurm script according to your needs (activate virtual env,
request more memory, more computation time, ...).
'''

import os, errno, sys

logdir = 'logs-trial/' # directory to save log files (where stdout is flushed)
try:
    os.makedirs(logdir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

alg_name = sys.argv[1]
n_trials = int(sys.argv[2])
env_list = sys.argv[3:]

for env_name in env_list:
    for trial in range(n_trials):
        run_name = alg_name + '_' + env_name + '_' + str(trial)

        text = """\
#!/bin/bash

# job name
#SBATCH -J job_name

# logfiles
#SBATCH -o """ + logdir + """stdout_""" + run_name + """\
#SBATCH -e """ + logdir + """stderr_""" + run_name + """\

# request computation time hh:mm:ss
#SBATCH -t 24:00:00

# request virtual memory in MB per core
#SBATCH --mem-per-cpu=1000

# nodes for a single job
#SBATCH -n 1

#SBATCH -C avx2
#SBATCH -c 4

# activate virtual env
module load intel python/3.6
source ../.bashrc
python3 -m """ + alg_name + """ """ + env_name + """ """ + str(trial) + """ """ + str(trial) + """\
    """

        text_file = open('r.sh', "w")
        text_file.write(text)
        text_file.close()

        os.system('sbatch r.sh')
        os.remove('r.sh')
