from typing import ClassVar, Optional

from brax.envs.base import PipelineEnv
from brax.io import image, torch
import gym
from gym import spaces
from gym.vector import utils
import jax
import numpy as np

from lib.edited_brax.gym_wrapper import *

    
class TorchWrapper(GymWrapper):
  """Wrapper that converts Jax tensors to PyTorch tensors."""

  def __init__(self, env: gym.Env, device: Optional[torch.Device] = None):
    """Creates a gym Env to one that outputs PyTorch tensors."""
    super().__init__(env)
    self.device = device

  def reset(self):
    obs = super().reset()
    return torch.jax_to_torch(obs, device=self.device)
  
  def reset_for_render(self):
    pipeline_state, obs = super().reset_for_render()
    return pipeline_state, torch.jax_to_torch(obs, device=self.device)

  def step(self, action):
    action = torch.torch_to_jax(action)
    obs, reward, done, info = super().step(action)
    obs = torch.jax_to_torch(obs, device=self.device)
    reward = torch.jax_to_torch(reward, device=self.device)
    done = torch.jax_to_torch(done, device=self.device)
    info = torch.jax_to_torch(info, device=self.device)
    return obs, reward, done, info
  
  def step_for_render(self, action):
    action = torch.torch_to_jax(action)
    pipeline_state, obs, reward, done, info = super().step_for_render(action)
    obs = torch.jax_to_torch(obs, device=self.device)
    reward = torch.jax_to_torch(reward, device=self.device)
    done = torch.jax_to_torch(done, device=self.device)
    info = torch.jax_to_torch(info, device=self.device)
    return pipeline_state, obs, reward, done, info