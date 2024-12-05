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

from lib.env import *
from lib.network import *
from lib.viz import *

import pickle

from jax.debug import breakpoint as jst
from pdb import set_trace as st

def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    print(ydata[-1])
    clear_output(wait=True)
    plt.xlim([0, train_func.keywords['num_timesteps']])
    plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.savefig('./rewards.png')


if __name__=='__main__':
    envs.register_environment('custom_humanoid', customHumanoid)
    env_name = "custom_humanoid" 
    backend = 'generalized' 
    #50_000_000,

    env = envs.get_environment(env_name=env_name,
                            backend=backend)
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

    # jit_env_reset = jax.jit(env.reset)
    # jit_env_step = jax.jit(env.step)
    # # jit_inference_fn = jax.jit(inference_fn)

    # rollout = []
    # rng = jax.random.PRNGKey(seed=1)
    # state = jit_env_reset(rng=rng)
    # jst()
    # for t in range(1000):
    #     rollout.append(state.pipeline_state)
    #     # act_rng, rng = jax.random.split(rng)
    #     # act, _ = jit_inference_fn(state.obs, act_rng)
    #     act=jax.random.normal(rng, (69,))
    #     act=(act-act.mean())/act.std()
    #     state = jit_env_step(state, act)

    # train_func=functools.partial(ppo.train,  num_timesteps=1, num_evals=10, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=128, batch_size=64, seed=1)

    train_func=functools.partial(ppo.train,  num_timesteps=1, num_evals=1, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048, batch_size=1024, seed=1)

    max_y = 13000
    min_y = 0
    xdata, ydata = [], []
    times = [datetime.now()]

    make_inference_fn, params, _ = train_func(environment=env, progress_fn=progress)

    # model.save_params('./weights/params.pkl', params)
    params = model.load_params('./weights_humanoid/params.pkl')
    inference_fn = make_inference_fn(params)

    env = envs.create(env_name=env_name, backend=backend)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
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

    create_interactive_rollout(env, rollout, jit_env_reset)

    # model.save_params('./weights/rollout_viz.pkl', rollout)
    # st.success("Simulation complete!") link_names


