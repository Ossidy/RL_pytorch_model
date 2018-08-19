from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np

import gym
from atari_util import PreprocessAtari
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from agent import Agent
from env_pool import EnvPool
from visualization import ImageGenerator
from train import *


def make_env():
    game_id="KungFuMasterDeterministic-v0"
    env = gym.make(game_id)
    env = PreprocessAtari(env, height=42, width=42,
                          crop = lambda img: img[60:-30, 15:],
                          color=False, n_frames=1)
    return env


def flatten_obs(unflattened_obs, n_parallel_games):
    obs = unflattened_obs[0]
    for i in range(1, n_parallel_games):
        obs = np.vstack((obs, unflattened_obs[i]))
    return obs


def train(agent, env_pool, niters, n_parallel_games, gamma, save_path=None, curiosity=False):

    n_parallel_games = n_parallel_games
    gamma = gamma
    
    if cuda:
        agent.cuda() 

    # pool = EnvPool(agent, make_env, n_parallel_games)

    opt_decision = torch.optim.Adam(list(agent.decisionUnit.parameters()) + list(agent.memoryUnit.parameters()), lr=1e-5)
    opt_curiosity = torch.optim.Adam(agent.curiosityUnit.get_all_params(), lr=1e-3)
    if curiosity:
        opts = [opt_decision, opt_curiosity]
    else:
        opts = [opt_decision]

    rewards_history = []
    ImageGen = ImageGenerator('./graph.png')

    for i in range(niters):  
        memory = list(pool.prev_memory_states)
        rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
        loss = train_on_rollout(agent, opts, rollout_obs, rollout_actions, rollout_rewards, rollout_mask, memory, gamma, curiosity)    
        
        if i % 100 == 0: 
            test_var = torch.autograd.Variable(torch.from_numpy(flatten_obs(rollout_obs, n_parallel_games)).cuda(), requires_grad=False)
            rewards_history.append(np.mean(evaluate(agent, env, n_games=1)))
            ImageGen(rewards_history)
            print(loss)

        # if i % 200 == 0 and save_path:
        #     agent.save_agent(save_path)
                
    return rewards_history



if __name__ == "__main__":
    n_parallel_games = 5
    gamma = 0.99

    env = make_env()

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print("Observation shape:", obs_shape)
    print("Num actions:", n_actions)
    print("Action names:", env.env.env.get_action_meanings())

    n_parallel_games = n_parallel_games
    gamma = gamma

    from agent import Agent
    agent = Agent(obs_shape, n_actions, n_parallel_games)

    if cuda:
        agent.cuda() 

        
    chkpt_dir = "./chkpt"

    pool = EnvPool(agent, make_env, n_parallel_games)

    train(agent, pool, 50000, n_parallel_games, gamma, save_path=chkpt_dir, curiosity=True)