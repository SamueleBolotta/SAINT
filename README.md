This repository implements MAPPO, a multi-agent variant of PPO that can join attention with other agents and employs a recurrent policy.
This repository is heavily based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt, https://github.com/google-research/google-research/tree/master/social_rl/multiagent_tfagents/joint_attention and https://github.com/dido1998/Recurrent-Independent-Mechanisms

## Environments supported:

- [Multiagent Particle-World Environments (MPEs)](https://github.com/openai/multiagent-particle-envs)

## 1. Usage

All core code is located within the onpolicy folder. The algorithms/ subfolder contains algorithm-specific code. 

* The envs/ subfolder contains environment wrapper implementations for the MPEs, SMAC, and Hanabi. 
* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 
* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 
* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 

More specifically:
* In onpolicy-algorithms-r_mappo-algorithm, there are two scripts:
- r_actor_critic contains the actor and the critic, with RIMs and Joint Attention
- rMAPPOPolicy is a wrapper for the actor and the critic,  which allows  to retrieve actions and values as well as to evaluate the actions

* In onpolicy-algorithms-r_mappo:
- the script r_mappo is the trainer class, which can calculate the value function loss, and crucially perform a training update using minibatch GD. It takes in the advantages from the buffer, it generates samples using a generator (either recurrent or feedforward) and performs a PPO update on each of those samples.

* In onpolicy-algorithms-utils:
- the script JOINT_ATTENTION contains the implementation of joint attention as well as a function that takes in attention maps from all agents and computes a bonus using JSD
- the script RIM contains the implementation of RIMs

* In onpolicy-envs, there are implementations of the MPEs.

* In onpolicy-runner-separated, there are two scripts:

- base_runner, which is the main runner. It takes in the wrapper for the actor  and the critic as well as the trainer class. It's where the handling of the multiple agents happens. It computes returns and trains the agents.
- mpe_runner, for each step of each episode, it executes the step function, computes the attention bonuses and inserts data into the buffer. Then, it computes returns and updates the network

## 2. Installation

Example installation. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
conda create -n marl python==3.6.2
conda activate marl
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -e .
pip install absl-py gym wandb tensorboardX torch_ac imageio pyglet PIL

```


### 2.3 Install MPE

``` Bash
# install this package first
pip install seaborn
```

### 3.Train

Here we use train_mpe.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

```

