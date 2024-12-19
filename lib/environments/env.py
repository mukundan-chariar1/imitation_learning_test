from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco

from lib.environments.utils import *

from jax.debug import breakpoint as jst
from pdb import set_trace as st

import torch

# def add_named_attribute_to_system(sys: base.System, attr_name: str, attr_value):
#     """Adds a custom attribute to the system."""
#     # Create a dictionary to hold the new attribute
#     custom_attribute = {attr_name: attr_value}

#     # Update the system with the new attribute
#     sys = sys.tree_replace(custom_attribute)

#     return sys

class customHumanoid(PipelineEnv):
    def __init__(self, **kwargs):

        path = 'lib/model/humanoid.xml'
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        sys = mjcf.load_model(mj_model)

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(sys=sys, **kwargs)

        self._healthy_reward=5.0
        self._upward_reward_weight=10
        self._vel_reward_weight=15

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = self.sys.init_q
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics={'counter': zero}
        return State(pipeline_state, obs, reward, done, metrics)
    
    def step(self, state: State, action: jax.Array) -> State:
        state_prev = state.pipeline_state
        count=state.metrics['counter']+1
        new_state=self.pipeline_step(state_prev, action)
        upward_reward=jp.where(new_state.x.pos[0, :] >= jp.ones(3)*0.5, new_state.x.pos[0, :]**2*self._upward_reward_weight, 0)[-1]
        vel_reward=(new_state.subtree_com[0, :]-state_prev.subtree_com[0, :])/self.dt
        vel_reward=jp.where(vel_reward>=jp.ones(3)*0.5, vel_reward*self._vel_reward_weight, 0)[-1]

        reward=vel_reward+upward_reward

        obs = self._get_obs(new_state, action)
        done=jp.where(new_state.x.pos[0, 2] >= jp.ones(1)*0.5, jp.zeros(1), jp.ones(1))[0]

        state.metrics.update(counter=count)
        return state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        position = pipeline_state.q
        velocity = pipeline_state.qd

        return jp.concatenate([position, velocity])
    
class SMPLHumanoid(PipelineEnv):
    def __init__(self, use_newton_solver=True, use_6d_notation=False, ignore_joint_positions=True, **kwargs):
        path = 'lib/model/smpl_humanoid_v2.xml'
        mj_model = mujoco.MjModel.from_xml_path(path)
        if use_newton_solver: mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        sys = mjcf.load_model(mj_model)

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(sys=sys, **kwargs)

        self._upright_reward_weight=0.1
        self._upward_reward_weight=5
        self._vel_reward_weight=15

        self._use_6d_notation=use_6d_notation
        self._ignore_joint_positions=ignore_joint_positions

        # add_named_attribute_to_system(self.sys, 'solo', [9, 12, 11, 10, 13, 0])
        # add_named_attribute_to_system(self.sys, 'left', [15, 1, 2, 14, 4, 17, 18, 3, 16])
        # add_named_attribute_to_system(self.sys, 'right', [20, 5, 6, 19, 8, 22, 23, 7, 21])

    def reset(self, rng: jax.Array) -> State:
        # Might consider swapping models here to keep consistency
        # domain randomization?
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = self.sys.init_q

        # qvel = jax.random.uniform(
        #     rng2, (self.sys.qd_size(),))
        qvel=jp.zeros(self.sys.qd_size(),)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics={'counter': zero}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        state_prev = state.pipeline_state
        count=state.metrics['counter']+1
        new_state=self.pipeline_step(state_prev, action)
        upward_reward=jp.where(new_state.x.pos[0, :] >= jp.ones(3)*0.5, new_state.x.pos[0, :]**2*self._upward_reward_weight, 0)[-1]

        vel_reward=(new_state.subtree_com[0, :]-state_prev.subtree_com[0, :])/self.dt
        vel_reward=jp.where(vel_reward>=jp.ones(3)*0.5, vel_reward*self._vel_reward_weight, 0)[-1]

        upright_reward=jp.where(jp.abs(quaternion_to_rotation_6d(new_state.x.rot[[0, 9, 10, 11, 12, 13], :])-quaternion_to_rotation_6d(jp.array([[1, 0, 0, 0]])))<=0.1, self._upright_reward_weight, 0).sum()

        reward=vel_reward+upward_reward+upright_reward

        obs = self._get_obs(new_state, action)

        done=jp.where(new_state.x.pos[0, 2] >= jp.ones(1)*0.5, jp.zeros(1), jp.ones(1))[0]

        state.metrics.update(counter=count)

        return state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self._use_6d_notation:
            obs=jp.concatenate([pipeline_state.x.pos, quaternion_to_rotation_6d(pipeline_state.x.rot), pipeline_state.xd.vel, pipeline_state.xd.ang], -1).flatten()
        else: 
            obs=jp.concatenate([pipeline_state.q, pipeline_state.qd])

        # jst()

        # might consider changing to maximal coords and r6d
        return obs
    

class SMPLHumanoid_imitator(PipelineEnv):
    def __init__(self, use_newton_solver=True, use_6d_notation=False, ignore_joint_positions=True, **kwargs):
        path = 'lib/model/smpl_humanoid_v2.xml'
        mj_model = mujoco.MjModel.from_xml_path(path)
        if use_newton_solver: mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        sys = mjcf.load_model(mj_model)

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(sys=sys, **kwargs)

        self._upright_reward_weight=0.1
        self._upward_reward_weight=5
        self._vel_reward_weight=15

        self._use_6d_notation=use_6d_notation
        self._ignore_joint_positions=ignore_joint_positions

        self.pelvis=jp.array(axis_angle_to_quaternion(torch.load('/home/mukundan/Desktop/Summer_SEM/imitation_learning/data/rep_00_output.pt')['full_pose'].reshape(-1, 24, 3)[:, 0]).numpy())
        self.rest=jp.array(torch.load('/home/mukundan/Desktop/Summer_SEM/imitation_learning/data/rep_00_output.pt')['full_pose'].reshape(-1, 24, 3)[:, 1:].numpy())

    def reset(self, rng: jax.Array) -> State:
        # Might consider swapping models here to keep consistency
        # domain randomization?
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = jp.concatenate([self.sys.init_q[:3], self.pelvis[0], self.rest[0].flatten()])

        # jst()

        # qvel = jax.random.uniform(
        #     rng2, (self.sys.qd_size(),))
        qvel=jp.zeros(self.sys.qd_size(),)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics={'counter': zero}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        state_prev = state.pipeline_state
        count=state.metrics['counter']+1
        new_state=self.pipeline_step(state_prev, action)
        upward_reward=jp.where(new_state.x.pos[0, :] >= jp.ones(3)*0.5, new_state.x.pos[0, :]**2*self._upward_reward_weight, 0)[-1]

        vel_reward=(new_state.subtree_com[0, :]-state_prev.subtree_com[0, :])/self.dt
        vel_reward=jp.where(vel_reward>=jp.ones(3)*0.5, vel_reward*self._vel_reward_weight, 0)[-1]

        upright_reward=jp.where(jp.abs(quaternion_to_rotation_6d(new_state.x.rot[[0, 9, 10, 11, 12, 13], :])-quaternion_to_rotation_6d(jp.array([[1, 0, 0, 0]])))<=0.1, self._upright_reward_weight, 0).sum()

        reward=vel_reward+upward_reward+upright_reward

        obs = self._get_obs(new_state, action)

        done=jp.where(new_state.x.pos[0, 2] >= jp.ones(1)*0.5, jp.zeros(1), jp.ones(1))[0]

        state.metrics.update(counter=count)

        return state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self._use_6d_notation:
            obs=jp.concatenate([pipeline_state.x.pos, quaternion_to_rotation_6d(pipeline_state.x.rot), pipeline_state.xd.vel, pipeline_state.xd.ang], -1).flatten()
        else: 
            obs=jp.concatenate([pipeline_state.q, pipeline_state.qd])

        # jst()

        # might consider changing to maximal coords and r6d
        return obs