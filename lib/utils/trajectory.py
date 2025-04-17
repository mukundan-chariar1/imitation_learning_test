import pickle
import jax
import jax.numpy as jnp
import brax
from brax import envs
import numpy as np
from typing import List, Dict, Any
import os
from flax import serialization
import functools
from brax.training.agents.ppo import train as ppo
from brax.io import model

from lib.environments.classic import *

from pdb import set_trace as st
from jax.debug import breakpoint as jst

def create_inference_function(params_path: str='weights/params_walker.pkl', 
                              env_name: str='humanoid',
                              backend: str='mjx'
                              ) -> callable:
    env = envs.get_environment(env_name=env_name,
                           backend=backend)

    train_func=functools.partial(ppo.train,  num_timesteps=1, num_evals=1, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=1, num_updates_per_batch=1, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1, batch_size=2, seed=1)

    make_inference_fn, params, _ = train_func(environment=env)
    params = model.load_params(params_path)
    inference_fn = make_inference_fn(params)

    return inference_fn

def save_rollouts(inference_fn: callable,
                  env_name: str='humanoid',
                  backend: str='mjx',
                  num_trajs: int=100,
                  save_folder: str='trajectories') ->  None:
    
    env = envs.get_environment(env_name=env_name,
                           backend=backend)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    for rollout_idx in range(num_trajs):
        if os.path.exists(os.path.join(save_folder, f"{rollout_idx}.pkl")):
            continue
        rollout = []
        rng = jax.random.PRNGKey(seed=rollout_idx)
        state = jit_env_reset(rng=rng)
        # for _ in range(1000):
        while state.done.item()<1:
            rollout.append(state.pipeline_state)
            act_rng, rng = jax.random.split(rng)
            act, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_env_step(state, act)

        with open(os.path.join(save_folder, f"{rollout_idx}.pkl"), 'wb') as f:  # 'wb' = write binary
            pickle.dump(rollout, f)

def load_rollouts(num_trajs: int=100,
                  save_folder: str='trajectories'):
    for rollout_idx in range(num_trajs):
        if not os.path.exists(os.path.join(save_folder, f"{rollout_idx}.pkl")):
            raise RuntimeError(f"{rollout_idx} does not exist")

        with open(os.path.join(save_folder, f"{rollout_idx}.pkl"), 'rb') as f:  # 'wb' = write binary
            rollout=pickle.load(f)

        yield rollout

def load_rollout(num_traj: int=0,
                 save_folder: str='trajectories') -> list:
    with open(os.path.join(save_folder, f"{num_traj}.pkl"), 'rb') as f:  # 'wb' = write binary
        rollout=pickle.load(f)

    return rollout

def get_observation(num_trajs: int=100,
                    save_folder: str='trajectories'):
    
    observations=[]
    # for rollout in load_rollouts(num_trajs, save_folder):
    for rollout_idx in range(num_trajs):
        q, qd=[], []
        rollout=load_rollout(rollout_idx)
        for r in rollout:
            q.append(r.q)
            qd.append(r.qd)
        q=jp.stack(q)
        qd=jp.stack(qd)
        observations.append(jp.concatenate((q, qd), axis=-1))
    return observations

# Example usage
if __name__ == "__main__":
    # inference_fn=create_inference_function()

    # save_rollouts(inference_fn, num_trajs=100)

    get_observation()