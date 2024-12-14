# rl-project-iql
Reproduction of the results mentioned in the paper "Offline Reinforcement Learning with Implicit Q-Learning" (https://arxiv.org/abs/2110.06169)

The code used in this repo was adapted from Ilya Kostrikov's repository: https://github.com/ikostrikov/implicit_q_learning

The data used to run these experiments was obtained from **d4rl**.

Average return plots, loss plots and tensorboard log files from our reproduction of the experiments can be found at: https://drive.google.com/drive/folders/1z4HHzVcW0MKEmprZj-aPRPTPlIxJ45aF?usp=sharing


Note that these instructions have been picked up as is from Ilya's repository.
## How to run the code 

### Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install --upgrade "jax[cuda]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

### Run training

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000
```