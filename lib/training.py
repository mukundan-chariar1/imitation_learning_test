import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from lib.env import *
from lib.network import *
from lib.viz import *
# from lib.edited_brax import gym_wrapper as _gym_wrapper
# from lib.edited_brax import torch_wrapper as _torch_wrapper
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

from jax.debug import breakpoint as jst
from pdb import set_trace as st

import gc

def make_custom_env():
    # Replace `customHumanoid` with your environment constructor
    return customHumanoid()

def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        discounted_rewards.insert(0, cumulative)
    return torch.tensor(discounted_rewards, dtype=torch.float32)


def train(env, policy_net, policy_optimizer, value_net, value_optimizer, num_envs=4, gamma=0.1, episodes=100):
    for episode in range(episodes):  # Number of episodes
        obs = env.reset()

        # Storage for trajectory data
        all_rewards = []
        all_log_probs = []
        all_values = []

        done=False


        while not done:
            # Sample actions from the policy
            action, log_prob = policy_net.select_action(obs, env)

            # for u in range(10):
            value = value_net(obs)
            obs, rewards, done, _ = env.step(action)
            if done: break

            all_rewards.append(rewards)
            all_log_probs.append(log_prob)
            all_values.append(value)

        # Convert lists to tensors
        rewards = torch.stack(all_rewards).transpose(0, 1)  # shape [num_envs, num_steps]
        log_probs = torch.stack(all_log_probs).transpose(0, 1)  # shape [num_envs, num_steps]
        values = torch.stack(all_values).transpose(0, 1)  # shape [num_envs, num_steps]

        # Calculate discounted returns and advantages
        discounted_returns = torch.zeros_like(rewards)

        for i in range(num_envs):
            discounted_returns[i] = compute_discounted_rewards(rewards[i].detach().cpu().numpy(), gamma)
        advantages = discounted_returns.unsqueeze(-1) - values.detach()

        # Policy Loss -(log_probs*rewards.unsqueeze(-1)).mean()
        policy_loss = -(log_probs * advantages).mean()

        # Value Loss
        value_loss = nn.functional.mse_loss(values, discounted_returns.unsqueeze(-1))

        # Backpropagation
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Logging
        avg_reward = rewards[:, :-1].sum(dim=1).mean().item()
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        torch.cuda.empty_cache()
        gc.collect()

def evaluate_policy(policy_net, env, num_steps=100):
    total_rewards = []
    obs = env.reset()
    done=False

    with torch.no_grad():
        while not done:
            action, log_prob = policy_net.select_action(obs, env)
            # for u in range(10):
            obs, rewards, done, _ = env.step(action)
            total_rewards.append(rewards.detach().cpu().numpy())
            if done: break
    
    # Calculate total rewards per episode
    total_rewards = np.array(total_rewards).sum(axis=0)
    return total_rewards

if __name__=='__main__':
    envs.register_environment('custom_humanoid', customHumanoid)
    num_envs=1
    # env = envs.create('custom_humanoid', 1000, batch_size=num_envs)
    env = envs.create('humanoid', 1000, batch_size=num_envs)
    env = gym_wrapper.GymWrapper(env)

    env = torch_wrapper.TorchWrapper(env, device='cuda')

    observation = env.reset() 
    input_size = env.observation_space.shape[-1]  # Example input size (could be different based on your environment)
    hidden_size = 128  # Hidden layer size
    output_size = env.action_space.shape[-1]  # Example output size (matching the number of actions in the Humanoid)

    mlp = MLP(input_size, hidden_size, output_size).to('cuda')
    value_fn = ValueFunction(input_size, hidden_size).to('cuda')
    optimizer_actor = optim.Adam(mlp.parameters(), lr=5e-3)
    optimizer_critic = optim.Adam(value_fn.parameters(), lr=5e-3)

    os.makedirs('./weights', exist_ok=True)

    if not (osp.exists('./weights/policy_net.pth') or osp.exists('./weights/value_net.pth')):
        train(env, mlp, optimizer_actor, value_fn, optimizer_critic, episodes=100, num_envs=num_envs)

        torch.save(mlp.state_dict(), './weights/policy_net.pth')
        torch.save(value_fn.state_dict(), './weights/value_net.pth')
    else:
        mlp.load_weights('./weights/policy_net.pth')
        value_fn.load_weights('./weights/value_net.pth')

    # average_rewards = []
    # for _ in range(2):
    #     rewards = evaluate_policy(mlp, env)
    #     average_rewards.append(np.mean(rewards))
    
    # # Print results
    # average_reward = np.mean(average_rewards)
    # print(f"Average reward over {2} episodes: {average_reward:.2f}")

    # viz_env = envs.create('custom_humanoid', 100)
    viz_env = envs.create('humanoid', 1000, batch_size=num_envs)
    viz_env = gym_wrapper.GymWrapper(viz_env)
    viz_env = torch_wrapper.TorchWrapper(viz_env, device='cuda')
    
    save_video_unroll(viz_env, mlp, 100, './rollout.mp4')

    # create_interactive_rollout(viz_env, mlp)