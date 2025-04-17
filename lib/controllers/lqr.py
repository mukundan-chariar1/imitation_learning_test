from configs import constants as _C

from typing import Optional, Tuple
from functools import partial

import brax
import jax.numpy as jp
import jax.scipy as jscipy
import numpy as np
from scipy.linalg import solve_discrete_are
from lib.utils.viz import *
from lib.environments.utils import *
from lib.environments.env import *
from lib.environments.scaling import *
from lib.utils.wrappers import *
from lib.controllers.utils import *
import functools
from brax import envs
from brax.envs import create

from lib.controllers.load_traj import get_traj_from_wham

import mujoco
import mujoco.mjx as mjx
import mediapy as media

from pdb import set_trace as st
from jax.debug import breakpoint as jst

import matplotlib.pyplot as plt

def fhlqr(A: np.array, 
          B: np.array, 
          Q: np.array, 
          R: np.array, 
          Qf: np.array, 
          N: int=100) -> tuple[list, list]:
    nx, nu = B.shape
    assert A.shape == (nx, nx), "A must be of shape (nx, nx)"
    assert Q.shape == (nx, nx), "Q must be of shape (nx, nx)"
    assert R.shape == (nu, nu), "R must be of shape (nu, nu)"
    assert Qf.shape == (nx, nx), "Qf must be of shape (nx, nx)"

    P = [np.zeros((nx, nx)) for _ in range(N)]
    K = [np.zeros((nu, nx)) for _ in range(N - 1)]

    P[-1] = Qf.copy()

    for k in range(N - 2, -1, -1):
        _Q=Q #if k%5==0 else Qf
        K[k] = jp.linalg.inv(R + B.T @ P[k + 1] @ B) @ (B.T @ P[k + 1] @ A)
        P[k] = _Q + A.T @ P[k + 1] @ A - A.T @ P[k + 1] @ B @ K[k]

    return P, K

def ihlqr(A: np.array, 
          B: np.array, 
          Q: np.array, 
          R: np.array, 
          tol: float = 1e-5, 
          verbose: bool = False, 
          freq: int = 1000) -> tuple[np.array, np.array]:
    nx, nu = B.shape
    P = Q.copy()

    i=0
    while True:
        if verbose and i%freq==0: print(f"Iteration : {i}")
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K

        if np.linalg.norm(P - P_new, ord=jp.inf) < tol:
            if verbose: print(f"Converged at iteration {i}")
            return P_new, K
        i+=1

        P = P_new

def get_gains(n_frames: int, 
              N: int=100, 
              model_loc: str='lib/model/smpl_humanoid_no_transl_v2.xml') -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    with open(model_loc, 'r') as f:
        xml = f.read()

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    model.opt.timestep=model.opt.timestep*n_frames

    offset=get_offset(model, data)
    data.qpos[2] += offset
    state_setpoint = data.qpos.copy()
    ctrl_setpoint=get_ctrl_setpoint(model, data)

    Q, R=get_costs(model, data, ctrl_setpoint)
    
    A, B=linearize_dynamics(model, data, ctrl_setpoint, state_setpoint)

    P, K=ihlqr(A, B, Q, R, verbose=True)
    # P, K=fhlqr(A, B, Q, R, Q.copy()*10, N)

    return A, B, Q, R, K

if __name__=='__main__':
    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    envs.register_environment('custom_humanoid', SMPLHumanoid)
    envs.register_environment('imitator_humanoid', SMPLHumanoid_imitator)
    env = create(env_name='custom_humanoid', backend='mjx')
    partial_randomization_fn = functools.partial(
        domain_randomize_no_vmap, env=env
        )
    randomization_fn = functools.partial(
        partial_randomization_fn, rng=sub_rng
        )
    wrapped_env=DomainRandomizationpWrapper(env, randomization_fn=randomization_fn)
    state = wrapped_env.reset(rng)

    rot, ang, transl, vel=get_traj_from_wham()
    N=rot.shape[0]

    A, B, Q, R, K=get_gains(wrapped_env._n_frames, N)

    K=jp.array(K)

    A=jp.array(A)
    B=jp.array(B)

    root_transl=state.pipeline_state.x.pos[0].copy()
    x=np.concatenate([np.stack([transl[:, 0], transl[:, 1], np.full(N, root_transl[-1])]).T, rot.reshape(N, -1)], axis=-1)
    x_dot=np.concatenate([vel.reshape(N, -1), ang.reshape(N, -1)], axis=-1)

    X_ref=np.zeros((N, state.pipeline_state.qd.shape[0]*2))
    for i in range(N):
        X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])]]=x[i, :]
        X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])+state.pipeline_state.qd.shape[0]]]=x_dot[i, :]

    # X_ref=jp.concatenate([x, x_dot], axis=-1)
    x0=X_ref[0]

    nq=wrapped_env.sys.nq
    nv=wrapped_env.sys.nv

    initialization=(x0[:x0.shape[0]//2], x0[x0.shape[0]//2:])

    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    env=SMPLHumanoid_imitator(backend='mjx')
    randomization_fn = functools.partial(
        partial_randomization_fn, rng=sub_rng
        )
    wrapped_env=DomainRandomizationpWrapper(env, randomization_fn=randomization_fn)
    state=wrapped_env.reset_(([], [], []))

    jit_env_reset = jax.jit(wrapped_env.reset_)
    jit_env_step = jax.jit(wrapped_env.step)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(initialization)

    # x_pos=quaternion_to_axis_angle(state.pipeline_state.x.rot.copy()).flatten()
    # x_ang=state.pipeline_state.xd.ang.copy().flatten()
    # x_transl=state.pipeline_state.x.pos.copy().flatten()[:3]
    # x_vel=state.pipeline_state.xd.vel.copy().flatten()[:3]
    # x_ref=jp.concatenate((x_transl, x_pos, x_vel, x_ang))

    X_sim = np.zeros((N, state.pipeline_state.qd.shape[0]*2))
    U_sim = np.zeros((N-1, nv), dtype=float)

    root_transl=state.pipeline_state.x.pos[0].copy()
    x=np.concatenate([np.stack([transl[:, 0], transl[:, 1], np.full(N, root_transl[-1])]).T, rot.reshape(N, -1)], axis=-1)
    x_dot=np.concatenate([vel.reshape(N, -1), ang.reshape(N, -1)], axis=-1)
    # X_ref=jp.concatenate([x, x_dot], axis=-1)
    X_ref=np.zeros((N, state.pipeline_state.qd.shape[0]*2))
    for i in range(N):
        X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])]]=x[i, :]
        X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])+state.pipeline_state.qd.shape[0]]]=x_dot[i, :]

    X_sim[0]=X_ref[0]

    for t in range(len(U_sim)):
        rollout.append(state.pipeline_state)
        # x_pos=quaternion_to_axis_angle(state.pipeline_state.x.rot.copy()).flatten()
        # x_ang=state.pipeline_state.xd.ang.copy().flatten()
        # x_transl=state.pipeline_state.x.pos.copy().flatten()[:3]
        # x_vel=state.pipeline_state.xd.vel.copy().flatten()[:3]
        # x=jp.concatenate((x_transl, x_pos, x_vel, x_ang))
        _q=state.pipeline_state.q.copy()
        q=jp.concatenate([_q[:3], quaternion_to_axis_angle(_q[3:7]), _q[7:]])
        qd=state.pipeline_state.qd.copy()
        x=jp.concatenate((q, qd))
        # act=-K @ (x - X_ref[t])
        act=-K[t] @ (x - X_ref[t])
        state = jit_env_step(state, act)

        X_sim[t+1]=x.copy()

    # plot_states_vs_reference_individual(X_sim, X_ref)

    # st()

    create_interactive_rollout(env=wrapped_env, rollout=rollout, headless=False)

    st()

    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    state = wrapped_env.reset(rng)
    state = wrapped_env.step(state, jp.zeros(wrapped_env.action_size))