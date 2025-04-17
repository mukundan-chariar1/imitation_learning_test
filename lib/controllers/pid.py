import jax
import jax.numpy as jnp
from brax import mjx
# from brax.physics.base import Motion

from configs import constants as _C

from typing import Optional, Tuple
from functools import partial

import jax.numpy as jp
import jax.scipy as jscipy
import numpy as np
from scipy.linalg import solve_discrete_are, solve

import functools
import brax
from brax import envs
from brax.envs import create
from brax.envs.base import PipelineEnv, State

from lib.controllers.load_traj import get_traj_from_wham, get_traj_from_pkl

from lib.utils.viz import *
from lib.environments.utils import *
from lib.environments.env import *
from lib.environments.scaling import *
from lib.utils.wrappers import *
from lib.controllers.utils import *
from lib.controllers.lqr import *

import mujoco
import mujoco.mjx as mjx
import mediapy as media

# from brax import mjx

from pdb import set_trace as st
from jax.debug import breakpoint as jst

import matplotlib.pyplot as plt

def pd_control(q_desired, qd_desired, q_current, qd_current, kp, kd):
    e_pos = q_desired - q_current
    e_vel = qd_desired - qd_current

    # Matrix PD control
    return kp @ e_pos + kd @ e_vel

def controller_step(state, q_desired, qd_desired, kp, kd):
    # Get joint positions and velocities
    q=format_state_forward(state.pipeline_state)
    q_current=q[:q.shape[0]//2]
    qd_current=q[q.shape[0]//2:]

    # Compute torques
    torques = pd_control(q_desired, qd_desired, q_current, qd_current, kp, kd)

    # # Apply torques using MJX step
    # new_state = env.step(state, torques)
    # return new_state

    return torques

if __name__=='__main__':
    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    # envs.register_environment('custom_humanoid', SMPLHumanoid)
    # envs.register_environment('imitator_humanoid', SMPLHumanoid_imitator)
    # env = create(env_name='imitator_humanoid', backend='mjx', n_frames=5)
    # env=SMPLHumanoid_imitator_old(backend='mjx', n_frames=5)
    env=SMPLHumanoid_imitator(backend='mjx', n_frames=5)
    partial_randomization_fn = functools.partial(
        domain_randomize_no_vmap, env=env
        )
    randomization_fn = functools.partial(
        partial_randomization_fn, rng=sub_rng
        )
    wrapped_env=DomainRandomizationpWrapper(env, randomization_fn=randomization_fn)
    # wrapped_env=env
    jit_env_reset = jax.jit(wrapped_env.reset_)
    # jit_env_reset=jax.jit(wrapped_env.reset)
    jit_env_step = jax.jit(wrapped_env.step)

    # display_init_positions_imitator(wrapped_env, headless=False)#, randomize=True)

    # st()

    rot, ang, transl, vel=get_traj_from_wham()
    # rot, ang, transl, vel=get_traj_from_pkl()
    N=rot.shape[0]

    st()

    nq, nu=wrapped_env.sys.nv*2, wrapped_env.sys.nu

    # root_transl=state.pipeline_state.x.pos[0].copy()
    root_transl=wrapped_env.initial_qpos[:3].copy()
    x=jp.concatenate([jp.stack([transl[:, 0], transl[:, 1], jp.full(N, root_transl[-1])]).T, rot.reshape(N, -1)], axis=-1)
    x_dot=jp.concatenate([vel.reshape(N, -1), ang.reshape(N, -1)], axis=-1)

    X_ref=jp.zeros((N, nq))
    for i in range(N):
        X_ref=X_ref.at[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])]].set(x[i, :])
        X_ref=X_ref.at[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])+nq//2]].set(x_dot[i, :])

    # X_ref=jp.concatenate([x, x_dot], axis=-1)
    x0=X_ref[0]
    st()

    initialization=(x0[:x0.shape[0]//2], x0[x0.shape[0]//2:])

    Kp = build_selection_matrix_from_indices() * 1e6
    Kd = build_selection_matrix_from_indices() * 0

    rollout=[]

    jit_controller_step=jax.jit(controller_step)

    # state = wrapped_env.reset(rng)  # or use an initial state
    state=jit_env_reset((None, None, None))
    # create_interactive_rollout(env=wrapped_env, rollout=[state.pipeline_state], headless=False)
    display_init_positions_imitator(env=wrapped_env, initialization=initialization, headless=False)

    st()
    
    for t in range(N):
        # jax.debug.print(f'at step {t}')
        rollout.append(state.pipeline_state)
        q_des = X_ref[t, :nq//2]
        qd_des = X_ref[t, nq//2:]
        act = jit_controller_step(state, q_des, qd_des, Kp, Kd)
        state=jit_env_step(state, act)



    create_interactive_rollout(env=wrapped_env, rollout=rollout, headless=False)

    st()