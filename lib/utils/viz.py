from typing import Optional

from brax.envs.base import Env
from brax.io import html

import jax
import jax.numpy as jp

import os
import webbrowser

from jax.debug import breakpoint as jst
from pdb import set_trace as st

def create_interactive_rollout(env: Env, rollout: list, height: Optional[int]=1080, headless: Optional[bool]=True, path: Optional[str]='temp.html') -> None:
    html_file = html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout, height)

    abs_path = os.path.abspath(path)

    with open(abs_path, 'w') as f:
        f.write(html_file)
    print(f"Rollout successful, saved at {abs_path}")
    if not headless: load_interactive_rollout(path)

def load_interactive_rollout(path: Optional[str]='temp.html') -> None:
    abs_path = os.path.abspath(path)
    url = 'file://' + abs_path
    webbrowser.open(url)

def test_environment_for_debug(env: Env, headless: Optional[bool]=True, path: Optional[str]='temp.html') -> None:
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)

    for t in range(1000):
        rollout.append(state.pipeline_state)
        act=jp.zeros(env.action_size)
        # act = -0.1 * jp.ones((69, ))
        state = jit_env_step(state, act)

    create_interactive_rollout(env, rollout, headless=headless, path=path)

def display_init_positions(env: Env, headless: Optional[bool]=True, path: Optional[str]='temp.html') -> None:
    jit_env_reset = jax.jit(env.reset)
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)

    create_interactive_rollout(env, [state.pipeline_state], headless=headless, path=path)