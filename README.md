#### Description
This is a small and simple collection of some reinforcement learning algorithms. The core idea of this repo is to have minimal structure, such that each algorithm is easy to understand and to modify. For this reason, each algorithm has a separate folder, independent from the others. Only approximators (neural network, linear functions, ...), policy classes, and auxiliary functions (for plotting or collecting data with gym-like environments) are shared.  

> Note that an algorithm can have different versions. For example, SPG can learn the critic by using Monte-Carlo estimates or by temporal difference.

The repository has a modular structure and no installation is needed. To run an algorithm, from the root folder execute  
`python3 -m <ALG>.<RUN_SCRIPT> <ENV_NAME> <SEED>`  
(seed is optional, default is 1). At each iteration, data about the most important statistics (average return, value function loss, entropy, ...) is saved in  
`data-trial/<ALG_NAME>/<ENV_NAME>/<DATE_TIME>.dat`.  
For example, running  
`python3 -m ddpg.ddpg Pendulum-v0 0`  
will generate  
`data-trial/ddpg/Pendulum-v0/180921_155842.dat`.  

You can also save/load the learned model and visualize the graph. For more info, check `demo.py`. The demo also shows how to use the LQR environment and how to plot value functions.  

Finally, use any of the `run` scripts in the root folder to run several trials of the same algorithm in parallel (see the scripts for instructions).


> Note that all scripts use [flexible memory](https://github.com/tensorflow/tensorflow/issues/1578), i.e.,
> ```
> config_tf = tf.ConfigProto()
> config_tf.gpu_options.allow_growth=True
> session = tf.Session(config=config_tf)
> ```


#### Requirements
* [`python 3.5+`](https://www.python.org/download/releases/3.0/)
* [`tensorflow 1.12.0+`](https://www.tensorflow.org/install/)
* [`tensorflow probability 0.5+`](https://www.tensorflow.org/probability/)
* [`gym 0.10+`](https://github.com/openai/gym/)
* [`numpy 1.15+`](https://docs.scipy.org/doc/numpy/user/install.html)
* [`scipy 1.2+`](https://www.scipy.org/install.html)
* [`matplotlib`](https://matplotlib.org/users/installing.html)

You can also use other physics simulators, such as [Roboschool](https://github.com/openai/roboschool/), [PyBullet](https://pypi.org/project/pybullet/) and [MuJoCo](https://github.com/openai/mujoco-py/).

#### Common files
* `approximators.py`   : neural network, random Fourier features, polynomial features
* `average_env.py`     : introduces state resets to consider average return MDPs
* `data_collection.py` : functions for sampling MDP transitions and getting mini-batches
* `filter_env.py`      : modifies a gym environment to have states and actions normalized in [-1,1]
* `logger.py`          : creates folders for saving data
* `noise.py`           : noise functions
* `plotting.py`        : to plot value functions
* `policy.py`          : implementation of Gaussian policy
* `rl_utils.py`        : RL functions, such as [generalized advantage estimation](https://arxiv.org/abs/1506.02438) and [Retrace](https://arxiv.org/pdf/1606.02647.pdf)

#### Custom environment
* `lqr_env.py`         : linear-quadratic regulator

#### Algorithm-specific files
* `solver.py`          : (optional) defines optimization routines required by the algorithm
* `hyperparameters.py` : defines the hyperparameters (e.g., number of transitions per iteration, network sizes and learning rates)
* `<NAME>.py`          : script to run the algorithm (e.g., `ppo.py` or `ddpg.py`)

#### Implemented algorithms
* Stochastic policy gradient (SPG). The folder includes [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) and two actor-critic versions.
* [Deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971).
* [Twin delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477).
* [Trust region policy optimization (TRPO)](https://arxiv.org/abs/1502.05477).
* [Proximal policy optimization (PPO)](https://arxiv.org/abs/1707.06347).
* [Relative entropy policy search (REPS)](http://jmlr.org/papers/v18/16-142.html).
* [Actor-critic REPS (AC-REPS)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12247).  

<br/>

* [TD-Regularized Actor-Critic Methods](xxx) (TD-REG and GAE-REG) is implemented for PPO and TRPO.
* [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) (ICM) is implemented for PPO.
