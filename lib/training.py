import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from lib.env import *
from lib.network import *
from lib.viz import *
import random

from brax import actuator
from brax import base
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco
import gym
from gym.vector import SyncVectorEnv

import os
import os.path as osp
from tqdm import tqdm

from collections import deque

from jax.debug import breakpoint as jst
from pdb import set_trace as st

import gc

def train(env, ac, optimizer):
    #track scores
    scores = []

    #track recent scores
    recent_scores = deque(maxlen = 100)

    pbar=tqdm(total=NUM_EPISODES, desc='training', leave=False)

    #run episodes
    for episode in range(NUM_EPISODES):
        
        #init variables
        state = env.reset()
        done = False
        score = 0
        I = 1
        
        #run episode, update online
        for step in range(MAX_STEPS):
            
            #get action and log probability
            action, lp = ac.select_action(state)
            
            #step with action
            new_state, reward, done, _ = env.step(action)
            
            #update episode score
            score += reward
            
            #get state value of current state
            state_val = ac.estimate_value(state)
            
            #get state value of next state
            new_state_val = ac.estimate_value(new_state)
            
            #if terminal state, next state val is 0
            if done:
                new_state_val = torch.tensor([0]).float().unsqueeze(0).to(DEVICE)
            
            #calculate value function loss with MSE
            val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
            val_loss *= I
            
            #calculate policy loss
            advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
            policy_loss = -lp * advantage
            policy_loss *= I

            loss=val_loss+policy_loss
            
            # #Backpropagate policy
            # policy_optimizer.zero_grad()
            # policy_loss.backward(retain_graph=True)
            # policy_optimizer.step()
            
            # #Backpropagate value
            # stateval_optimizer.zero_grad()
            # val_loss.backward()
            # stateval_optimizer.step()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if done:
                break
                
            #move into new state, discount I
            state = new_state
            I *= DISCOUNT_FACTOR
        
        #append episode score 
        scores.append(score.detach().cpu())
        recent_scores.append(score)   

        pbar.set_postfix_str(f"average scores {torch.stack(scores).mean():.2f}")
        pbar.update(1)

if __name__=='__main__':
    DISCOUNT_FACTOR = 0.99

    #number of episodes to run
    NUM_EPISODES = 1000

    #max steps per episode
    MAX_STEPS = 10000

    #device to run model on 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    envs.register_environment('custom_humanoid', customHumanoid)
    num_envs=1
    env = envs.create('custom_humanoid', 1000, batch_size=num_envs, backend='generalized')
    env = gym_wrapper.GymWrapper(env)

    env = torch_wrapper.TorchWrapper(env, device=DEVICE)

    observation = env.reset() 
    input_size = env.observation_space.shape[-1]  # Example input size (could be different based on your environment)
    hidden_size = 128  # Hidden layer size
    output_size = env.action_space.shape[-1]  # Example output size (matching the number of actions in the Humanoid)

    ac = ActorCritic(input_size, hidden_size, output_size).to(DEVICE)
    optimizer = optim.Adam(ac.parameters(), lr=5e-3)

    os.makedirs('./weights', exist_ok=True)

    # if not (osp.exists('./weights/policy_net.pth') or osp.exists('./weights/value_net.pth')):
    train(env, ac, optimizer)

    # else:
    #     mlp.load_weights('./weights/policy_net.pth')
    #     value_fn.load_weights('./weights/value_net.pth')

    # average_rewards = []
    # for _ in range(2):
    #     rewards = evaluate_policy(mlp, env)
    #     average_rewards.append(np.mean(rewards))
    
    # # Print results
    # average_reward = np.mean(average_rewards)
    # print(f"Average reward over {2} episodes: {average_reward:.2f}")

    # viz_env = envs.create('custom_humanoid', 100)
    viz_env = envs.create('custom_humanoid', 1000, batch_size=num_envs, backend='generalized')
    viz_env = gym_wrapper.GymWrapper(viz_env)
    viz_env = torch_wrapper.TorchWrapper(viz_env, device='cuda')
    
    save_video_unroll(viz_env, ac, 100, './rollout.mp4')

    # create_interactive_rollout(viz_env, mlp)