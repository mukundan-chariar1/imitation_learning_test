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

from lib.utils.math import *

import mujoco
import mujoco.mjx as mjx
import mediapy as media

from pdb import set_trace as st
from jax.debug import breakpoint as jst

with open('/home/mukundan/Desktop/Summer_SEM/imitation_learning/lib/model/smpl_humanoid_v5.xml', 'r') as f:
  xml = f.read()

rng = jax.random.PRNGKey(0)
rng, sub_rng, key = jax.random.split(rng, 3)

envs.register_environment('custom_humanoid', SMPLHumanoid)

env = create(env_name='custom_humanoid', backend='mjx', use_6d_notation=True)
partial_randomization_fn = functools.partial(
      domain_randomize_no_vmap, env=env
    )
randomization_fn = functools.partial(
      partial_randomization_fn, rng=sub_rng
    )
wrapped_env=DomainRandomizationpWrapper(env, randomization_fn=randomization_fn)
state = env.reset(rng)

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

# Compute the matrix exponential of Dc * dt
D = jscipy.linalg.expm(Dc * env.dt)

# Extract submatrices A and B
A = D[:nx, :nx]  # A is the top-left nx x nx block
B = D[:nx, nx:nx+nu]

Q=jp.eye(nx)
R=jp.eye(nu)

Qf=10*jp.eye(nx)

def fhlqr(
  A: jax.Array,  # State transition matrix (nx, nx)
  B: jax.Array,  # Control input matrix (nx, nu)
  Q: jax.Array,  # State cost matrix (nx, nx)
  R: jax.Array,  # Control cost matrix (nu, nu)
  Qf: jax.Array,  # Terminal state cost matrix (nx, nx)
  N: int=100  # Horizon size
  ) -> tuple[list, list]:
  """
  Finite-horizon Linear Quadratic Regulator (LQR) solver.

  Args:
  A: State transition matrix (nx, nx).
  B: Control input matrix (nx, nu).
  Q: State cost matrix (nx, nx).
  R: Control cost matrix (nu, nu).
  Qf: Terminal state cost matrix (nx, nx).
  N: Horizon size.

  Returns:
  P: List of cost-to-go matrices (nx, nx) for each time step.
  K: List of feedback gain matrices (nu, nx) for each time step.
  """
  # Check sizes of everything
  nx, nu = B.shape
  assert A.shape == (nx, nx), "A must be of shape (nx, nx)"
  assert Q.shape == (nx, nx), "Q must be of shape (nx, nx)"
  assert R.shape == (nu, nu), "R must be of shape (nu, nu)"
  assert Qf.shape == (nx, nx), "Qf must be of shape (nx, nx)"

  # Instantiate P and K
  P = [jp.zeros((nx, nx)) for _ in range(N)]  # Cost-to-go matrices
  K = [jp.zeros((nu, nx)) for _ in range(N - 1)]  # Feedback gain matrices

  # Initialize P[N-1] with Qf
  P[-1] = Qf.copy()

  # Riccati recursion
  for k in range(N - 2, -1, -1):
    K[k] = jp.linalg.inv(R + B.T @ P[k + 1] @ B) @ (B.T @ P[k + 1] @ A)
    P[k] = Q + A.T @ P[k + 1] @ A - A.T @ P[k + 1] @ B @ K[k]

  return P, K

def ihlqr(
    A: jax.Array,  # State transition matrix (nx, nx)
    B: jax.Array,  # Control input matrix (nx, nu)
    Q: jax.Array,  # State cost matrix (nx, nx)
    R: jax.Array,  # Control cost matrix (nu, nu)
    max_iter: int = 1000,  # Maximum iterations for Riccati
    tol: float = 1e-5  # Convergence tolerance
) -> tuple[jax.Array, jax.Array]:
    """
    Infinite-horizon Linear Quadratic Regulator (LQR) solver using JAX NumPy.

    Args:
        A: State transition matrix (nx, nx).
        B: Control input matrix (nx, nu).
        Q: State cost matrix (nx, nx).
        R: Control cost matrix (nu, nu).
        max_iter: Maximum number of iterations for Riccati recursion.
        tol: Convergence tolerance.

    Returns:
        P: Cost-to-go matrix (nx, nx).
        K: Feedback gain matrix (nu, nx).

    Raises:
        RuntimeError: If the Riccati recursion does not converge within max_iter.
    """
    # Get size of x and u from B
    nx, nu = B.shape

    # Initialize P with Q
    P = Q.copy()

    # Riccati recursion
    for riccati_iter in range(max_iter):
        # Compute feedback gain matrix K
        K = jp.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        
        # Update cost-to-go matrix P
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K

        # Check for convergence
        if jp.linalg.norm(P - P_new, ord=jp.inf) < tol:
            return P_new, K

        P = P_new

    # If max_iter is reached without convergence
    raise RuntimeError("ihlqr did not converge")


# Step 2: Compute control setpoints

P, K=fhlqr(A, B, Q, R, Qf)

st()

P_new, K=ihlqr(A, B, Q, R)

st()

rng = jax.random.PRNGKey(0)
rng, sub_rng, key = jax.random.split(rng, 3)

state = env.reset(rng)
state = env.step(state, jp.zeros(env.action_size))