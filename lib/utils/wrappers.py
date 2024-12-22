from typing import Callable, Optional
import functools

from brax.base import System
from brax.envs.base import Env, State, Wrapper

import jax
from jax import numpy as jp

from jax.debug import breakpoint as jst
from pdb import set_trace as st

class DomainRandomizationpWrapper(Wrapper):
  """Wrapper for domain randomization."""

  def __init__(
      self,
      env: Env,
      randomization_fn: Callable[[System], System],
  ):
    super().__init__(env)
    self._sys_v = randomization_fn(self.sys)

  def _env_fn(self, sys: System) -> Env:
    env = self.env
    env.unwrapped.sys = sys
    return env

  def reset(self, rng: jax.Array) -> State:
    def reset(sys, rng):
      env = self._env_fn(sys=sys)
      return env.reset(rng)
    state = reset(self._sys_v, rng)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def step(sys, s, a):
      env = self._env_fn(sys=sys)
      return env.step(s, a)

    res = step(
        self._sys_v, state, action
    )
    return res
  
class CustomDomainRandomizationpWrapper(Wrapper):
  """Wrapper for domain randomization."""

  def __init__(
      self,
      env: Env,
      randomization_fn: Callable[[System, jax.Array], System],
      rng: jax.Array,
  ):
    super().__init__(env)
    self._sys_v = randomization_fn(self.sys, rng)
    self._randomization_fn=functools.partial(randomization_fn, sys=self.sys)

  def _re_randomize(self, rng: jax.Array):
    return self._randomization_fn(rng=rng)

  def _env_fn(self, sys: System) -> Env:
    env = self.env
    env.unwrapped.sys = sys
    return env

  def reset(self, rng: jax.Array) -> State:
    def reset(sys, rng):
      env = self._env_fn(sys=sys)
      return env.reset(rng)
    
    rng, rng1, rng2=jax.random.split(rng, 3)
    self._sys_v=self._re_randomize(rng=rng1)
    state = reset(self._sys_v, rng2)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def step(sys, s, a):
      env = self._env_fn(sys=sys)
      return env.step(s, a)

    res = step(
        self._sys_v, state, action
    )
    return res