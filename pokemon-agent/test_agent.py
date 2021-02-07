from poke_env.player.random_player import RandomPlayer
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from gen8Agent import SimpleRLPlayer
from metricLogger import MetricLogger
from pokeTrainer import Trainer

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
env_player = SimpleRLPlayer(battle_format="gen8randombattle")
# print(env_player.action_space)
test_trainer = Trainer(state_dim=10, hidden_dim=1000, action_dim=18, save_dir=save_dir)

logger = MetricLogger(save_dir)

def dqn_training(game, trainer, episodes, log):

    for e in range(episodes):
        game.reset()
        last_battle = list(game.battles.keys())[-1]
        state = game.embed_battle(game.battles[last_battle])
        while True:
            action = trainer.act(state)
            next_state, reward, done, info = game.step(action)
            trainer.cache(state, next_state, action, reward, done)
            q, loss = trainer.learn()
            logger.log_step(reward, loss, q)
            state = next_state
            if done:
                break
        logger.log_episode()
        if e % 20 == 0:
            logger.record(episode=e, epsilon=trainer.exploration_rate, step=trainer.curr_step)

opponent = RandomPlayer(battle_format="gen8randombattle")

env_player.play_against(
    env_algorithm=dqn_training,
    opponent=opponent,
    env_algorithm_kwargs={"trainer": test_trainer, "episodes": 1000, "log": logger},
)


# def dqn_training(player, dqn, nb_steps):
#     dqn.fit(player, nb_steps=nb_steps)
#
#     # This call will finished eventual unfinshed battles before returning
#     player.complete_current_battle()
#
# opponent = RandomPlayer(battle_format="gen8randombattle")
#
# # Training
# env_player.play_against(
#     env_algorithm=dqn_training,
#     opponent=opponent,
#     env_algorithm_kwargs={"dqn": dqn, "nb_steps": 100000},
# )
#
