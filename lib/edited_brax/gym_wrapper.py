from typing import ClassVar, Optional

from brax.envs.base import PipelineEnv
from brax.io import image, torch
import gym
from gym import spaces
from gym.vector import utils
import jax
import numpy as np


class GymWrapper(gym.Env):
  """A wrapper that converts Brax Env to one that follows Gym API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
               env: PipelineEnv,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    # self.metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 1 / self._env.dt
    # }
    self.seed(seed)
    self.backend = backend
    self._state = None
    # self._sys=env.sys

    obs = np.inf * np.ones(self._env.observation_size, dtype='float32')
    self.observation_space = spaces.Box(-obs, obs, dtype='float32')

    action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
    self.action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')

    def reset(key):
      key1, key2 = jax.random.split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    # We return device arrays for pytorch users.
    return obs

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    # We return device arrays for pytorch users.
    return obs, reward, done, info
  
  def step_for_render(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)

    pipeline_state=self._state.pipeline_state
    # We return device arrays for pytorch users.
    return pipeline_state, obs, reward, done, info
  
  def reset_for_render(self):
    self._state, obs, self._key = self._reset(self._key)
    # We return device arrays for pytorch users.

    pipeline_state=self._state.pipeline_state
    return pipeline_state, obs

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def render(self, mode='human'):
    if mode == 'rgb_array':
      sys, state = self._env.sys, self._state
      if state is None:
        raise RuntimeError('must call reset or step before rendering')
      return image.render_array(sys, state.pipeline_state, 256, 256)
    else:
      return super().render(mode=mode)  # just raise an exception