import flax
import brax
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
import functools
import jax
import os

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output

# import streamlit as stream

# from lib.environments.env import *
from lib.network import *
from lib.utils.viz import *
from lib.utils.wrappers import *
from lib.environments.env import *
from lib.environments.scaling import *

import pickle

from jax.debug import breakpoint as jst
from pdb import set_trace as st

# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)
# jax.config.update("jax_enable_x64", True)

def format_elapsed_time(seconds):
    """Converts elapsed seconds to HH:MM:SS format"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    current_time = datetime.now()
    elapsed_seconds = (current_time - start_time).total_seconds()  # Compute elapsed time in seconds
    elapsed_time_str = format_elapsed_time(elapsed_seconds)
    print(f"{elapsed_time_str} - {ydata[-1]}")
    # clear_output(wait=True)
    # plt.xlim([0, train_fn.keywords['num_timesteps']])
    ax.clear()
    ax.set_xlabel('# Environment Steps')
    ax.set_ylabel('Reward per Episode')
    ax.set_ylim(0, max_y)
    ax.plot(xdata, ydata, marker='o', linestyle='-', label="Training Progress")

    # Update legend
    ax.legend()
    fig.savefig("./rewards.png")

if __name__=='__main__':
    # st()
    envs.register_environment('custom_humanoid', SMPLHumanoid)
    # envs.register_environment('custom_humanoid', customHumanoid)
    # envs.register_environment('custom_humanoid', SkelHumanoid)
    env_name = "custom_humanoid" 
    backend = 'mjx' 

    env = envs.get_environment(env_name=env_name,
                            backend=backend, use_6d_notation=False)
    display_init_positions(env, headless=False)
    # test_environment_for_debug(env)
    # test_environment_for_debug(env, jp.array([1]))
    # exit()
    # st()

    # train_func=functools.partial(ppo.train,  num_timesteps=50_000_000, num_evals=10, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048, batch_size=1024, seed=1)

    partial_randomization_fn = functools.partial(
          domain_randomize, env=env
        )

    train_func=functools.partial(ppo.train,  num_timesteps=10_000, num_evals=2, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=1, num_updates_per_batch=1, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1, batch_size=2, seed=1, randomization_fn=partial_randomization_fn)

    max_y = 13000
    min_y = 0
    xdata, ydata = [], []
    start_time=datetime.now()
    times = [start_time]

    fig, ax = plt.subplots(figsize=(8, 5))  
    ax.set_xlabel('# Environment Steps')
    ax.set_ylabel('Reward per Episode')

    make_inference_fn, params, _ = train_func(environment=env, progress_fn=progress)

    model.save_params('./weights/params.pkl', params)
    params = model.load_params('./weights/params.pkl')
    inference_fn = make_inference_fn(params)

    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    viz_partial_randomization_fn = functools.partial(
          domain_randomize_no_vmap, env=env
        )
    
    randomization_fn = functools.partial(
          viz_partial_randomization_fn, rng=sub_rng
        )
    viz_env = envs.create(env_name=env_name, backend=backend, use_6d_notation=False)
    wrapped_env=DomainRandomizationpWrapper(viz_env, randomization_fn=randomization_fn)

    jit_env_reset = jax.jit(wrapped_env.reset)
    jit_env_step = jax.jit(wrapped_env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    for t in range(1000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        # act=jp.zeros(env.action_size)
        state = jit_env_step(state, act)

    create_interactive_rollout(env=wrapped_env, rollout=rollout, headless=False)


