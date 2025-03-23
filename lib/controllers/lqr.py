from typing import Optional

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

from lib.controllers.load_traj import get_traj

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

rng = jax.random.PRNGKey(0)
rng, sub_rng, key = jax.random.split(rng, 3)

envs.register_environment('custom_humanoid', SMPLHumanoid)
envs.register_environment('imitator_humanoid', SMPLHumanoid_imitator)
env = create(env_name='custom_humanoid', backend='mjx', use_6d_notation=True)
partial_randomization_fn = functools.partial(
      domain_randomize_no_vmap, env=env
    )
randomization_fn = functools.partial(
      partial_randomization_fn, rng=sub_rng
    )
wrapped_env=DomainRandomizationpWrapper(env, randomization_fn=randomization_fn)
state = wrapped_env.reset(rng)

# test_environment_for_debug(wrapped_env, headless=False, randomize=False)

x_pos=quaternion_to_axis_angle(state.pipeline_state.x.rot.copy()).flatten()
x_vel=state.pipeline_state.xd.ang.copy().flatten()
x=jp.concatenate((x_pos, x_vel))

nx=x.shape[0]
nu=x_vel.shape[0]

Ac=jp.concatenate((jp.concatenate((jp.zeros((nu, nu)), jp.eye((nu))), axis=-1), jp.zeros((nu, nx))))
Bc=jp.concatenate((jp.zeros((nu, nu)), jp.eye((nu))))

Dc = jp.block([
    [Ac, Bc],
    [jp.zeros((nu, nx+nu))]
])

# D = jscipy.linalg.expm(Dc * (wrapped_env.dt/5))
D=jscipy.linalg.expm(Dc*wrapped_env.dt)

A = D[:nx, :nx]
B = D[:nx, nx:nx+nu]

Q=jp.block([[jp.eye(nu), jp.zeros((nu, nu))], [jp.zeros((nu, nu)), jp.zeros((nu, nu))]])
R=jp.zeros((nu, nu))

Qf=10*jp.eye(nx)

def fhlqr(A: jax.Array, B: jax.Array, Q: jax.Array, R: jax.Array, Qf: jax.Array, N: int=100) -> tuple[list, list]:
    nx, nu = B.shape
    assert A.shape == (nx, nx), "A must be of shape (nx, nx)"
    assert Q.shape == (nx, nx), "Q must be of shape (nx, nx)"
    assert R.shape == (nu, nu), "R must be of shape (nu, nu)"
    assert Qf.shape == (nx, nx), "Qf must be of shape (nx, nx)"

    P = [jp.zeros((nx, nx)) for _ in range(N)]
    K = [jp.zeros((nu, nx)) for _ in range(N - 1)]

    P[-1] = Qf.copy()

    for k in range(N - 2, -1, -1):
        K[k] = jp.linalg.inv(R + B.T @ P[k + 1] @ B) @ (B.T @ P[k + 1] @ A)
        P[k] = Q + A.T @ P[k + 1] @ A - A.T @ P[k + 1] @ B @ K[k]

    return P, K

def ihlqr(A: jax.Array, B: jax.Array, Q: jax.Array, R: jax.Array, max_iter: int = 1000, tol: float = 1e-5) -> tuple[jax.Array, jax.Array]:
    nx, nu = B.shape
    P = Q.copy()

    for riccati_iter in range(max_iter):
        K = jp.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K

        if jp.linalg.norm(P - P_new, ord=jp.inf) < tol:
            return P_new, K

        P = P_new
    raise RuntimeError(f"ihlqr did not converge, tol: {jp.linalg.norm(P - P_new, ord=jp.inf):.8f}")

rot, ang=get_traj()
N=rot.shape[0]

# P, K=fhlqr(A, B, Q, R, Qf, N)
P_new, K=ihlqr(A, B, Q, R)

X_ref=jp.concatenate((rot.reshape((N, -1)), ang.reshape(N, -1)), axis=-1)
x0=X_ref[0]

X_sim = [jp.zeros(nx) for _ in range(N)]  # Simulated states
U_sim = [jp.zeros(nu) for _ in range(N-1)]  # Simulated control inputs

X_sim[0] = x0

for i in range(N-1):
    # U_sim[i] = jp.clip(-K[i] @ (X_sim[i] - X_ref[i]), u_min, u_max)
    # U_sim[i]=-K[i] @ (X_sim[i] - X_ref[i])
    U_sim[i]=-K @ (X_sim[i] - X_ref[i])
    X_sim[i+1] = A @ X_sim[i] + B @ U_sim[i]

X_sim=jp.stack(X_sim)
initialization=(X_sim[0, :X_sim.shape[1]//2], X_sim[0, X_sim.shape[1]//2:])

# plot_states_vs_reference_individual(X_sim, X_ref)

# st()

rng = jax.random.PRNGKey(0)
rng, sub_rng, key = jax.random.split(rng, 3)

# env = create(env_name='imitator_humanoid', backend='mjx', use_6d_notation=True, action_repeat=1)
# partial_randomization_fn = functools.partial(
#       domain_randomize_no_vmap, env=env
#     )

env=SMPLHumanoid_imitator()
randomization_fn = functools.partial(
      partial_randomization_fn, rng=sub_rng
    )
wrapped_env=DomainRandomizationpWrapper(env, randomization_fn=randomization_fn)
# state = wrapped_env.reset_(initialization)
state=wrapped_env.reset_(([], [], []))

jit_env_reset = jax.jit(wrapped_env.reset_)
jit_env_step = jax.jit(wrapped_env.step)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(initialization)

x_pos=quaternion_to_axis_angle(state.pipeline_state.x.rot.copy()).flatten()
x_vel=state.pipeline_state.xd.ang.copy().flatten()
x_ref=jp.concatenate((x_pos, x_vel))

for t in range(len(U_sim)):
    rollout.append(state.pipeline_state)
    x_pos=quaternion_to_axis_angle(state.pipeline_state.x.rot.copy()).flatten()
    x_vel=state.pipeline_state.xd.ang.copy().flatten()
    x=jp.concatenate((x_pos, x_vel))
    # act=U_sim[i][3:].copy()
    # act=-K @ (x - X_ref[i])
    act=-K @ (x - x_ref)
    state = jit_env_step(state, act[3:])

create_interactive_rollout(env=wrapped_env, rollout=rollout, headless=False)

st()

# rng = jax.random.PRNGKey(0)
# rng, sub_rng, key = jax.random.split(rng, 3)

# state = wrapped_env.reset(rng)
# state = wrapped_env.step(state, jp.zeros(wrapped_env.action_size))