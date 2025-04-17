from configs import constants as _C

from typing import Optional
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

def plot_states_vs_reference_individual(X_sim, X_ref):
    X_sim = np.array(X_sim)
    X_ref = np.array(X_ref)

    nx = X_sim.shape[1]
    N = X_sim.shape[0]

    time = np.arange(N)

    for i in range(nx):
        plt.figure(figsize=(10, 4))
        plt.plot(time, X_sim[:, i], label=f'X_sim[{i}]', marker='o')
        plt.plot(time, X_ref[:, i], label=f'X_ref[{i}]', linestyle='--', marker='x')
        plt.xlabel('Time Step')
        plt.ylabel(f'State {i}')
        plt.title(f'State {i}: Simulated vs Reference')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_states_vs_reference(X_sim, X_ref):
    X_sim = np.array(X_sim)
    X_ref = np.array(X_ref)

    nx = X_sim.shape[1]
    N = X_sim.shape[0]

    time = np.arange(N)

    fig, axes = plt.subplots(nx, 1, figsize=(10, 2 * nx))

    if nx == 1:
        axes = [axes]

    for i in range(nx):
        axes[i].plot(time, X_sim[:, i], label=f'X_sim[{i}]', marker='o')
        axes[i].plot(time, X_ref[:, i], label=f'X_ref[{i}]', linestyle='--', marker='x')
        axes[i].set_ylabel(f'State {i}')
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Step')

    plt.tight_layout()
    plt.show()

def trimmed_state_step(env, state, action):
    new_state = env.step(state, action)
    return format_state_forward(new_state.pipeline_state)

def format_state_forward(pipeline_grad):
    _q = pipeline_grad.q.copy()
    q = jp.concatenate([_q[:3], quaternion_to_axis_angle(_q[3:7]), _q[7:]])
    qd = pipeline_grad.qd.copy()
    return jp.concatenate((q, qd))

def format_state_backward(state):
    _q=state[:state.shape[0]//2].copy()
    q=jp.concatenate([_q[:3], axis_angle_to_quaternion(_q[3:6]), _q[6:]])
    qd=state[state.shape[0]//2:].copy()

    return q, qd

def linearize_dynamics(env, state, action, eps=1e-6):
    state_vector = format_state_forward(state.pipeline_state)
    n_states = state_vector.shape[0]
    n_actions = action.shape[0]

    def f_state(s_vec):
        q, qd = format_state_backward(s_vec)
        new_ps = state.pipeline_state.replace(q=q, qd=qd)
        new_state = state.replace(pipeline_state=new_ps)
        return trimmed_state_step(env, new_state, action)

    def f_action(a):
        return trimmed_state_step(env, state, a)

    def compute_A_col(i):
        basis_vec = jp.zeros(n_states).at[i].set(1.0)
        _, df = jax.jvp(f_state, (state_vector,), (basis_vec,))
        return df

    A = jax.vmap(compute_A_col)(jp.arange(n_states)).T

    def compute_B_col(i):
        basis_vec = jp.zeros(n_actions).at[i].set(1.0)
        _, df = jax.jvp(f_action, (action,), (basis_vec,))
        return df

    B = jax.vmap(compute_B_col)(jp.arange(n_actions)).T

    return A, B

def build_selection_matrix_from_indices(actuated_qs=_C.CONTROL.ROT_JNT_IDX[3:], total_dofs=98):
    """
    Build a (total_dofs x n_actuated) selection matrix J
    such that: J.T @ q_full == q_actuated
    """
    n_actuated = actuated_qs.shape[0]
    J = jp.zeros((total_dofs, n_actuated))
    J = J.at[actuated_qs, jp.arange(n_actuated)].set(1.0)
    return J.T