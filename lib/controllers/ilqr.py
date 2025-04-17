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

from lib.controllers.load_traj import get_traj_from_wham

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

def stage_cost(Q: jp.ndarray, 
               R: jp.ndarray, 
               Xref: jp.ndarray, 
               Uref: jp.ndarray, 
               x: jp.ndarray, 
               u: jp.ndarray, 
               k: int) -> float:
    """Stage cost at time step k"""
    dx = x-Xref[k]
    du = u-Uref[k]
    return (dx.T @ Q @ dx + du.T @ R @ du) / 2

def term_cost(Qf: jp.ndarray, 
              Xref: jp.ndarray, 
              x: jp.ndarray) -> float:
    """Terminal cost"""
    dx = x - Xref[-1]
    return (dx.T @ Qf @ dx) / 2

def stage_cost_expansion(Q: jp.ndarray, 
                         R: jp.ndarray, 
                         Xref: jp.ndarray, 
                         Uref: jp.ndarray, 
                         x: jp.ndarray, 
                         u: jp.ndarray, 
                         k: int) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """Stage cost expansion"""
    dJdx2 = Q
    dJdx = Q @ (x-Xref[k])
    dJdu2 = R
    dJdu = R @ (u-Uref[k])
    return dJdx2, dJdx, dJdu2, dJdu

def term_cost_expansion(Qf: jp.ndarray, 
                        Xref: jp.ndarray, 
                        x: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    """Terminal cost expansion"""
    dJndx2 = Qf
    dJndx = Qf @ (x - Xref[-1])
    return dJndx2, dJndx

def backward_pass(A: jp.ndarray, 
                  B: jp.ndarray, 
                  Q: jp.ndarray, 
                  R: jp.ndarray, 
                  Qf: jp.ndarray, 
                  nx: int, 
                  nu: int, 
                  N: int, 
                  Xref: jp.ndarray, 
                  Uref: jp.ndarray, 
                  X: jp.ndarray, 
                  U: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray, float]:
    """Compute the iLQR backwards pass given a dynamically feasible trajectory X and U"""
    
    # Initialize vectors for recursion
    P = [jp.zeros((nx, nx)) for _ in range(N)]   # cost to go quadratic term
    p = [jp.zeros(nx) for _ in range(N)]         # cost to go linear term
    d = [jp.zeros(nu) for _ in range(N-1)]       # feedforward control
    K = [jp.zeros((nu, nx)) for _ in range(N-1)] # feedback gain
    
    # Initialize terminal cost
    dJndx2, dJndx = term_cost_expansion(Qf, Xref, X[-1])
    P[-1] = dJndx2
    p[-1] = dJndx
    
    dJ = 0.0
    
    for i in range(N-2, -1, -1):
        # Compute Jacobians (A and B matrices)
        # Note: In Python, you'll need to implement or use a numerical differentiation function
        # Here we assume FD.jacobian is implemented elsewhere
        # A = FD.jacobian(lambda dx: discrete_dynamics(params, dx, U[i], params['model']['dt']), X[i])
        # B = FD.jacobian(lambda du: discrete_dynamics(params, X[i], du, params['model']['dt']), U[i])
        
        dJdx2, dJdx, dJdu2, dJdu = stage_cost_expansion(Q, R, Xref, Uref, X[i], U[i], i)
        
        G_x = dJdx + A.T @ p[i+1]
        G_u = dJdu + B.T @ p[i+1]
        G_xx = dJdx2 + A.T @ P[i+1] @ A
        G_uu = dJdu2 + B.T @ P[i+1] @ B
        # G_uu_bar = dJdu2 + B.T @ (P[i+1] + 1e-6 * jp.eye(nx)) @ B
        # G_uu_bar=G_uu+1e-6*jp.eye(nu)
        G_xu = A.T @ P[i+1] @ B
        G_ux = B.T @ P[i+1] @ A

        H = jp.block([[G_xx, G_xu],
                      [G_ux, G_uu]])
        
        beta=1e-6

        while jp.abs(jp.linalg.det(H))<1e-12:
            G_xx += beta * jp.eye(G_xx.shape[0])
            G_uu += beta * jp.eye(G_uu.shape[0])

            H = jp.block([[G_xx, G_xu],
                          [G_ux, G_uu]])

            beta *= 2
            # print(f"Applied regularization β={beta:.1e}")

        # if jp.abs(jp.linalg.det(G_uu))<1e-12: jst()
        
        # Solve for feedforward and feedback terms
        # d[i] = jp.linalg.solve(G_uu, G_u)
        # K[i] = jp.linalg.solve(G_uu, G_ux)

        d[i] = jp.linalg.inv(G_uu) @ G_u
        K[i] = jp.linalg.inv(G_uu) @ G_ux
        
        # Update cost-to-go terms
        P[i] = G_xx + K[i].T @ G_uu @ K[i] - G_xu @ K[i] - K[i].T @ G_ux
        p[i] = G_x - K[i].T @ G_u + K[i].T @ G_uu @ d[i] - G_xu @ d[i]
        
        dJ += (d[i].T @ G_uu @ d[i] + d[i].T @ G_u) / 2
    
    return d, K, dJ

def trajectory_cost(Q: jp.ndarray, 
                    R: jp.ndarray, 
                    Qf: jp.ndarray, 
                    Xref: jp.ndarray, 
                    Uref: jp.ndarray, 
                    N: int, 
                    X: jp.ndarray, 
                    U: jp.ndarray) -> float:
    """Compute the trajectory cost for trajectory X and U"""
    J = jp.float64(0.0)
    
    for i in range(N-1):
        J += stage_cost(Q, R, Xref, Uref, X[i], U[i], i)
        if jp.isnan(J): st()
    J += term_cost(Qf, Xref, X[-1])
    
    return J

def forward_pass(A: jp.ndarray, 
                 B: jp.ndarray, 
                 Q: jp.ndarray, 
                 R: jp.ndarray, 
                 Qf: jp.ndarray, 
                 nx: int, 
                 nu: int, 
                 N: int, 
                 Xref: jp.ndarray, 
                 Uref: jp.ndarray, 
                 X: jp.ndarray, 
                 U: jp.ndarray, 
                 d: jp.ndarray, 
                 K: jp.ndarray, 
                 max_linesearch_iters: int = 50,
                 jit_env_reset : Optional[Callable] =  None,
                 jit_env_step: Optional[Callable] = None,
                 state: Optional[State] = None,
                 initialization: Optional[tuple] = None) -> Tuple[jp.ndarray, jp.ndarray, float, float]:
    """Forward pass in iLQR with linesearch"""
    
    # Initial step length
    alpha = 1.0
    
    for _ in range(max_linesearch_iters):
        Xn = [jp.zeros(nx) for _ in range(N)]
        Un = [jp.zeros(nu) for _ in range(N-1)]
        Xn[0] = X[0].copy()

        state=jit_env_reset(initialization)
        
        for j in range(N-1):
            Un[j] = U[j] - alpha * d[j] - K[j] @ (Xn[j] - X[j])
            if not (jit_env_reset and jit_env_step and state): 
                X[j+1] = A@Xn[j]+B@Un[j]
            else:
                state=jit_env_step(state, Un[j])
                _q=state.pipeline_state.q.copy()
                q=jp.concatenate([_q[:3], quaternion_to_axis_angle(_q[3:7]), _q[7:]])
                qd=state.pipeline_state.qd.copy()

                Xn[j+1]=jp.concatenate((q, qd))
        
        J_new = trajectory_cost(Q, R, Qf, Xref, Uref, N, Xn, Un)
        
        if J_new < trajectory_cost(Q, R, Qf, Xref, Uref, N, X, U):
            return Xn, Un, J_new, alpha
        
        alpha *= 0.5
    
    raise RuntimeError("Forward pass failed to find a good step size")

def iLQR(A: jp.ndarray, 
         B: jp.ndarray, 
         Q: jp.ndarray, 
         R: jp.ndarray, 
         Qf: jp.ndarray, 
         nx: jp.ndarray, 
         nu: jp.ndarray, 
         N: jp.ndarray, 
         Xref: jp.ndarray, 
         Uref: jp.ndarray,
         x0: jp.ndarray,
         U: jp.ndarray,
         atol: float = 1e-3,
         max_iters: int = 250,
         verbose: bool = True,
         jit_env_reset : Optional[Callable] =  None,
         jit_env_step: Optional[Callable] = None,
         state: Optional[State] = None,
         initialization: Optional[tuple] = None) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """iLQR solver given an initial condition x0 and initial controls U"""
    
    # Initial rollout
    X = [jp.zeros(nx) for _ in range(N)]
    X[0] = x0.copy()
    state=jit_env_reset(initialization)
    for i in range(N-1):
        if not (jit_env_reset and jit_env_step and state): 
            X[i+1] = A@X[i]+B@U[i]
        else:
            state=jit_env_step(state, U[i])
            _q=state.pipeline_state.q.copy()
            q=jp.concatenate([_q[:3], quaternion_to_axis_angle(_q[3:7]), _q[7:]])
            qd=state.pipeline_state.qd.copy()

            X[i+1]=jp.concatenate((q, qd))
    
    J = trajectory_cost(Q, R, Qf, Xref, Uref, N, X, U)
    
    for ilqr_iter in range(1, max_iters+1):
        # try:
        d, K, dJ = backward_pass(A, B, Q, R, Qf, nx, nu, N, Xref, Uref, X, U)
        X, U, J, alpha = forward_pass(A, B, Q, R, Qf, nx, nu, N, Xref, Uref, X, U, d, K, jit_env_reset=jit_env_reset, jit_env_step=jit_env_step, state=state, initialization=initialization)
        # except Exception as e:
        #     raise RuntimeError(f"iLQR failed at iteration {ilqr_iter}: {str(e)}")
            # return X, U, K
        
        # Termination criteria
        if dJ < atol:
            if verbose:
                print("iLQR converged")
                print(f"{ilqr_iter:3d}   {J:10.3e}  {dJ:9.2e}  {dmax:9.2e}  {alpha:6.4f}")
            return X, U, K
        
        # Logging
        if verbose:
            dmax = max(jp.linalg.norm(di) for di in d)
            if (ilqr_iter - 1) % 10 == 0:
                print("iter     J           ΔJ        |d|         α         ")
                print("-------------------------------------------------")
            print(f"{ilqr_iter:3d}   {J:10.3e}  {dJ:9.2e}  {dmax:9.2e}  {alpha:6.4f}")
    
    raise RuntimeError("iLQR failed to converge")
    # return X, U, K

def get_gains_old(n_frames: int, 
              N: int=100, 
              model_loc: str='lib/model/smpl_humanoid_no_transl_v2.xml') -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    with open(model_loc, 'r') as f:
        xml = f.read()

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    nq, nu=model.nv*2, model.nu

    model.opt.timestep=model.opt.timestep*n_frames

    offset=get_offset(model, data)
    data.qpos[2] += offset
    state_setpoint = data.qpos.copy()
    ctrl_setpoint=get_ctrl_setpoint(model, data)

    Q, R=get_costs(model, data, ctrl_setpoint)
    Qf=Q*10
    
    A, B=linearize_dynamics(model, data, ctrl_setpoint, state_setpoint)

    P, K=ihlqr(A, B, Q, R, verbose=True)

    return A, B, Q, R, Qf, K, nq, nu

def get_gains(env: PipelineEnv, 
              N: int=100, 
              model_loc: str='lib/model/smpl_humanoid_no_transl_v2.xml') -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    Q_pos=jp.eye(env.sys.nv)

    Q_pos=Q_pos.at[_C.CONTROL.TRANSL_JNT_IDXS, :].multiply(0)
    Q_pos=Q_pos.at[:, _C.CONTROL.TRANSL_JNT_IDXS].multiply(0)

    Q = np.block([[Q_pos, jp.zeros((env.sys.nv, env.sys.nv))],
                [jp.zeros((env.sys.nv, 2*env.sys.nv))]])
    
    R=jp.eye(env.sys.nu)*1e-6
    Qf=Q*10

    state=wrapped_env.reset(rng)
    # perturbed_state = state.replace(
    # pipeline_state=state.pipeline_state.replace(
    #     q=state.pipeline_state.q + 1e-4*jp.ones_like(state.pipeline_state.q),
    #     qd=1e-4*jp.ones_like(state.pipeline_state.qd)
    # )
    # )
    action=jp.zeros(env.sys.nu).astype(jp.float32)

    sys=env.sys
    # mass_matrix=mjx.generalized_mass_matrix(sys, state.pipeline_state)
    # j = mjx.jacobian(sys, state.pipeline_state)
    
    A, B=linearize_dynamics(wrapped_env, state, action)

    jst()

    P, K=ihlqr(A, B, Q, R, verbose=True)

    jst()

    return A, B, Q, R, Qf, K, nq, nu

def test_quaternion_conversions():
    test_cases = [
        jp.array([1.0, 0.0, 0.0, 0.0]),        # No rotation
        jp.array([0.707, 0.707, 0.0, 0.0]),    # 90° X
        jp.array([0.0, 1.0, 0.0, 0.0]),        # 180° X
        jp.array([0.5, 0.5, 0.5, 0.5]),        # 120° XYZ
        jp.array([0.999, 0.001, 0.002, 0.0]),  # Tiny rotation
    ]
    
    for q in test_cases:
        q = q / jp.linalg.norm(q)  # Normalize
        aa = quaternion_to_axis_angle(q)
        q_recovered = axis_angle_to_quaternion(aa)
        
        print("\nOriginal:", q)
        print("Recovered:", q_recovered)
        print("Equivalent:", jp.allclose(q, q_recovered) or jp.allclose(q, -q_recovered))
        print("Norm preserved:", jp.allclose(jp.linalg.norm(q), jp.linalg.norm(q_recovered)))

if __name__=='__main__':
    test_quaternion_conversions()
    st()
    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    envs.register_environment('custom_humanoid', SMPLHumanoid)
    envs.register_environment('imitator_humanoid', SMPLHumanoid_imitator)
    env = create(env_name='custom_humanoid', backend='mjx', n_frames=5)
    # env=SMPLHumanoid(backend='mjx', n_frames=5)
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

    A, B, Q, R, Qf, K, nq, nu=get_gains(wrapped_env)

    K=jp.array(K)

    A=jp.array(A)
    B=jp.array(B)

    root_transl=state.pipeline_state.x.pos[0].copy()
    x=jp.concatenate([jp.stack([transl[:, 0], transl[:, 1], jp.full(N, root_transl[-1])]).T, rot.reshape(N, -1)], axis=-1)
    x_dot=jp.concatenate([vel.reshape(N, -1), ang.reshape(N, -1)], axis=-1)

    X_ref=jp.zeros((N, nq))
    for i in range(N):
        # X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])]]=x[i, :]
        # X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])+nq//2]]=x_dot[i, :]
        X_ref=X_ref.at[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])]].set(x[i, :])
        X_ref=X_ref.at[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])+nq//2]].set(x_dot[i, :])

    # X_ref=jp.concatenate([x, x_dot], axis=-1)
    x0=X_ref[0]

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

    X_sim = jp.zeros((N//100, nq))
    U_sim = jp.zeros((N//100-1, nu), dtype=float)

    Uref=U_sim.copy()

    root_transl=state.pipeline_state.x.pos[0].copy()
    x=jp.concatenate([jp.stack([transl[:, 0], transl[:, 1], jp.full(N, root_transl[-1])]).T, rot.reshape(N, -1)], axis=-1)
    x_dot=jp.concatenate([vel.reshape(N, -1), ang.reshape(N, -1)], axis=-1)
    # X_ref=jp.concatenate([x, x_dot], axis=-1)
    X_ref=jp.zeros((N, nq))
    for i in range(N):
        # X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])]]=x[i, :]
        # X_ref[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])+nq//2]]=x_dot[i, :]
        X_ref=X_ref.at[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])]].set(x[i, :])
        X_ref=X_ref.at[i, [jp.concatenate([_C.CONTROL.ROOT_TRANSL, _C.CONTROL.ROT_JNT_IDX])+nq//2]].set(x_dot[i, :])

    X_sim=X_sim.at[0].set(X_ref[0])

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
        act=-K @ (x - X_ref[t])
        # act=-K[t] @ (x - X_ref[t])
        U_sim=U_sim.at[t].set(act)
        state = jit_env_step(state, act)

        X_sim=X_sim.at[t+1].set(x)

    state = jit_env_reset(initialization)
    X, U, K=iLQR(A, B, Q, R, Qf, nq, nu, N//100, X_ref, Uref, X_ref[0], U_sim, jit_env_reset=jit_env_reset, jit_env_step=jit_env_step, state=state, initialization=initialization)
    # plot_states_vs_reference_individual(X_sim, X_ref)

    # st()

    create_interactive_rollout(env=wrapped_env, rollout=rollout, headless=False)

    st()

    rng = jax.random.PRNGKey(0)
    rng, sub_rng, key = jax.random.split(rng, 3)

    state = wrapped_env.reset(rng)
    state = wrapped_env.step(state, jp.zeros(wrapped_env.action_size))