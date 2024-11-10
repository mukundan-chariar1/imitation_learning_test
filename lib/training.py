import torch
import torch.nn as nn
import torch.optim as optim

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

import numpy as np

def train_step(observation, reward, action):
    """Perform a training step based on observation, action, and reward."""
    optimizer.zero_grad()  # Clear previous gradients
    action_pred = mlp(torch.tensor(observation, dtype=torch.float32))  # Predict action
    loss = ((action_pred - torch.tensor(action)) ** 2).mean()  # Simple MSE loss
    loss.backward()  # Backpropagate gradients
    optimizer.step()  # Update weights using optimizer
    return loss.item()

# Initialize the MLP

# Initialize optimizer

# Example loop interacting with the environment
def run_environment(env, mlp, n_steps=100):
    state = env.reset(jax.random.PRNGKey(0))  # Reset environment and get initial state
    observation = state.obs  # Get the initial observation

    for _ in range(n_steps):
        action = mlp.select_action(observation)  # Get action from MLP
        state = env.step(state, action)  # Take a step in the environment
        observation = state.obs  # Update the observation from the environment
        print("Reward:", state.reward)  # Print the reward at each step

# Example of how to run it in your environment
# Assuming you have the `Humanoid` environment defined:
# env = Humanoid()
# run_environment(env)

if __name__=='__main__':
    envs.register_environment('custom_humanoid', customHumanoid)
    env = envs.create('custom_humanoid', 100, batch_size=4)
    env = gym_wrapper.VectorGymWrapper(env)
    env = torch_wrapper.TorchWrapper(env, device='cuda')

    # batch_size = 32  # Number of experiences per batch
    # observations = torch.randn(batch_size, input_size)

    observation = env.reset() 
    input_size = env.observation_space.shape[-1]  # Example input size (could be different based on your environment)
    hidden_size = 64  # Hidden layer size
    output_size = env.action_space.shape[-1]  # Example output size (matching the number of actions in the Humanoid)

    mlp = MLP(input_size, hidden_size, output_size).to('cuda')
    value_fn = ValueFunction(input_size, hidden_size).to('cuda')
    optimizer_actor = optim.Adam(mlp.parameters(), lr=1e-3)
    optimizer_critic = optim.Adam(value_fn.parameters(), lr=1e-3)

    target_value=torch.zeros((4, 1)).to('cuda')

    # observation = state.obs  # Initial observation

    for step in range(100):  # Run for 100 steps
        # observation=np.array(observation)
        action, log_probs = mlp.select_action(observation, env)  # Get the action from the MLP
        value=value_fn(observation)

        observation, reward, done, _ = env.step(action)  # Step the environment with the selected action

        # import pdb; pdb.set_trace()



        target_value+=reward.reshape(4, 1)

        loss=target_value-value

        # observation = state.obs  # Update the observation
        # print("loss:", loss)  # Print the reward at each step

    # envs.register_environment('custom_humanoid', Humanoid)
    _env = envs.create('custom_humanoid', 100)
    _env = gym_wrapper.GymWrapper(_env)
    _env = torch_wrapper.TorchWrapper(_env, device='cuda')
    
    save_video_unroll(_env, mlp, 100, './rollout.mp4')