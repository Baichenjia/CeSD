# Constrained Ensemble Exploration for Unsupervised Skill Discovery

This is the official codebase for "Constrained Ensemble Exploration for Unsupervised Skill Discovery," accepted by **ICML 2024**, which utilizes an ensemble of skills in exploration, where each skill performs partition exploration and state-distribution constraints based on the prototypes.

This codebase is built on top of the [Unsupervised Reinforcement Learning Benchmark (URLB) codebase](https://github.com/rll-research/url_benchmark). Our method `CeSD` is implemented in `agents/cesd.py`, and the config is specified in `agents/cesd.yaml`. CeSD is based on the ensemble version of DDPG with ensemble critic, as implemented in `agents/ensemble_ddpg.py` and `agents/ensemble_ddpg.yaml`.

## Run CeSD

To pre-train CeSD, run the following command:

``` sh
python pretrain.py agent=cesd domain=walker
```

This script will produce several agent snapshots after training for `2M` frames and snapshots will be stored in `./models/${domain}_${now:%m.%d.%H.%M.%S}/${agent.name}`. 

To finetune CeSD, run the following command:

```sh
python finetune.py agent=cesd task=walker_stand snapshot_base_dir=YOUR_PATH_FILE
```

This will load a snapshot stored in `exp_local/YOUR_PATH_FILE/models/snapshot_2000000.pt`, initialize `Ensemble DDPG` with it (both the actor and critic), and start training on `walker_stand` using the extrinsic reward of the task.

## Requirements

We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with
```sh
conda activate urlb
```

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |

### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in the form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```

### BibTeX
```
@inproceedings{cesd2024,
  title={Constrained Ensemble Exploration for Unsupervised Skill Discovery},
  author={Bai, Chenjia and Yang, Rushuai and Zhang, Qiaosheng and Xu, Kang and Chen, Yi and Xiao, Ting and Li, Xuelong},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```
