from configs import constants as _C
import pickle
import joblib
from typing import Optional

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


import jax
from jax import numpy as jp

import mujoco

from lib.environments.utils import *

from jax.debug import breakpoint as jst
from pdb import set_trace as st

class SMPLHumanoid_basic(PipelineEnv):
    """
    uses SMPL formatted humanoid, makes it jump
    """
    def __init__(self, use_newton_solver: Optional[bool]=True, use_6d_notation: Optional[bool]=False, ignore_joint_positions: Optional[bool]=True, **kwargs):
        path = 'lib/model/smpl_sim_humanoid.xml'
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

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = self.sys.init_q
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

        vel_reward=(new_state.x.pos[0, :]-state_prev.x.pos[0, :])/self.dt
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
        return obs
    

class SMPLHumanoid_imitator_old(PipelineEnv):
    """
    Uses smpl formatted humanoid, adds imitation step to retrieve SMPL pose from body model
    """
    def __init__(self, use_newton_solver: Optional[bool]=True, use_6d_notation: Optional[bool]=False, ignore_joint_positions: Optional[bool]=True, **kwargs):
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

        # with open('dataset/data/data.pickle', 'rb') as file:
        #     data=pickle.load(file)['full_pose']
        data=joblib.load('test_data/drone_video/wham_output_modified.pkl')[0]['pose_world']
        self.pelvis=jp.array(axis_angle_to_quaternion(data.reshape(-1, 24, 3)[100:, 0]))
        self.rest=jp.array(data.reshape(-1, 24, 3)[100:, 1:])

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
    
class SMPLHumanoid(PipelineEnv):

    # TODO:
    #   - fix issues with scaling
    #   - tune PID controller values for position actuators
    #   - fix setting each position to initial value in step

    def __init__(self, use_newton_solver: Optional[bool]=True, use_6d_notation: Optional[bool]=False, ignore_joint_positions: Optional[bool]=True, **kwargs):
        path = 'lib/model/smpl_humanoid_v6.xml'
        mj_model = mujoco.MjModel.from_xml_path(path)
        if use_newton_solver: mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        sys = mjcf.load_model(mj_model)

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(sys=sys, **kwargs)

        self._upright_reward_weight=0.2
        self._upward_reward_hip_weight=4
        self._upward_reward_head_weight=5
        self._vel_reward_weight=15

        self._use_6d_notation=use_6d_notation
        self._ignore_joint_positions=ignore_joint_positions

        self.initial_geoms=sys.geom_size
        self.initial_qpos=sys.init_q
        self.initial_body_pos=sys.body_pos
        self.initial_mass=sys.body_mass
        self.initial_joint_lim=sys.jnt_range

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos=self.sys.init_q
        qvel=jp.zeros(self.sys.qd_size(),)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics={'counter': zero,}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        state_prev = state.pipeline_state
        count=state.metrics['counter']+1
        new_state=self.pipeline_step(state_prev, action)
        upward_reward=jp.where(new_state.x.pos[0, :] >= jp.ones(3)*0.5, new_state.x.pos[0, :]**2*self._upward_reward_hip_weight, 0)[-1]+jp.where(new_state.x.pos[13, :] >= jp.ones(3)*1, new_state.x.pos[0, :]**2*self._upward_reward_head_weight, 0)[-1]

        vel_reward=(new_state.subtree_com[[9, 10, 11, 12, 13], -1]-state_prev.subtree_com[[9, 10, 11, 12, 13], -1])/self.dt
        vel_reward=jp.where(vel_reward>jp.zeros((5, )), vel_reward*self._vel_reward_weight, -1).sum()

        vel_reward=(new_state.subtree_com[[0], -1]-state_prev.subtree_com[[0], -1])/self.dt
        vel_reward=jp.where(vel_reward>jp.zeros((1, )), vel_reward*self._vel_reward_weight**2, -1).sum()

        upright_reward=jp.where(jp.abs(quaternion_to_rotation_6d(new_state.x.rot[[0, 9, 10, 11, 12, 13], :])-quaternion_to_rotation_6d(jp.array([[1, 0, 0, 0]])))<=0.1, self._upright_reward_weight, 0).sum()

        reward=vel_reward+upward_reward+upright_reward

        obs = self._get_obs(new_state, action)

        done=jp.where(new_state.x.pos[0, 2] >= jp.ones(1)*0.5, jp.zeros(1), jp.ones(1))[0]

        state.metrics.update(counter=count)

        return state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done
        )
    
    def _convert_obs(self, pipeline_state: base.State,) -> jax.Array:
        temp=pipeline_state.q.copy()
        root_transl=temp[0:3].copy()
        root_rot=quaternion_to_rotation_6d(temp[3:7].copy())
        transl_jnts=temp[_C.INDEXING.TRANSL_JNT_IDXS].copy()
        rot_jnts=temp[_C.INDEXING.ROT_JNT_IDX].copy().reshape((-1, 3))
        converted_rot_jnts=axis_angle_to_rotation_6d(rot_jnts).flatten()

        q=jp.zeros(root_transl.shape[0]+root_rot.shape[0]+converted_rot_jnts.shape[0]+23)
        q=q.at[0:3].set(root_transl)
        q=q.at[3:9].set(root_rot)
        q=q.at[_C.INDEXING.R6D_TRANSL_IDXS].set(transl_jnts)
        q=q.at[_C.INDEXING.R6D_ROT_IDXS].set(converted_rot_jnts)

        return jp.concatenate([q, pipeline_state.qd])

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self._use_6d_notation:
            obs=self._convert_obs(pipeline_state=pipeline_state)
        else: 
            obs=jp.concatenate([pipeline_state.q, pipeline_state.qd])
        return obs
    
class SMPLHumanoid_imitator(PipelineEnv):

    # TODO:
    #   - convert x and xd to q and qd
    #   - may be in quaternion/ other format for q
    #   - axis angle in qd
    #   - q and qd may be local, figure out how to convert it

    def __init__(self, use_newton_solver: Optional[bool]=True, use_6d_notation: Optional[bool]=False, ignore_joint_positions: Optional[bool]=True, **kwargs):
        path = 'lib/model/smpl_humanoid_v6.xml'
        mj_model = mujoco.MjModel.from_xml_path(path)
        if use_newton_solver: mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        sys = mjcf.load_model(mj_model)

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(sys=sys, **kwargs)

        # self.n_frames=kwargs['n_frames']
        self._upright_reward_weight=0.2
        self._upward_reward_hip_weight=4
        self._upward_reward_head_weight=5
        self._vel_reward_weight=15

        self._use_6d_notation=use_6d_notation
        self._ignore_joint_positions=ignore_joint_positions

        self.initial_geoms=sys.geom_size
        self.initial_qpos=sys.init_q
        self.initial_body_pos=sys.body_pos
        self.initial_mass=sys.body_mass
        self.initial_joint_lim=sys.jnt_range

    def reset(self, rng):
        return self.reset_((None, None, None))

    def reset_(self, initialization: tuple) -> State:
        if len(initialization)==2:
            qpos=jp.zeros_like(self.sys.init_q)
            root_transl=self.sys.init_q[0:3].copy()
            qpos=qpos.at[0:3].set(root_transl)
            # root_rot=axis_angle_to_quaternion(initialization[0][:3].copy()+jp.array([jp.pi, 0, 0]))
            root_rot=axis_angle_to_quaternion(initialization[0][:3].copy())
            qpos=qpos.at[3:7].set(root_rot)
            transl_jnts=self.sys.init_q[_C.INDEXING.TRANSL_JNT_IDXS].copy()
            qpos=qpos.at[_C.INDEXING.TRANSL_JNT_IDXS].set(transl_jnts)
            rot_joints=initialization[0][_C.INDEXING.ROT_JNT_IDX-1].copy()
            qpos=qpos.at[_C.INDEXING.ROT_JNT_IDX].set(rot_joints)

            qvel=jp.zeros(self.sys.qd_size(),)
            qvel=qvel.at[3:6].set(initialization[1][:3].copy())
            qvel=qvel.at[_C.INDEXING.ROT_JNT_IDX-1].set(initialization[1][_C.INDEXING.ROT_JNT_IDX-1].copy())
        else:
            qpos=self.sys.init_q
            qvel=jp.zeros(self.sys.qd_size(),)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics={'counter': zero,}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        state_prev = state.pipeline_state
        count=state.metrics['counter']+1
        new_state=self.pipeline_step(state_prev, action)
        
        obs = self._get_obs(new_state, action)

        state.metrics.update(counter=count)

        return state.replace(
            pipeline_state=new_state, obs=obs, reward=0.0, done=0.0
        )
    
    def _convert_obs(self, pipeline_state: base.State,) -> jax.Array:
        temp=pipeline_state.q.copy()
        root_transl=temp[0:3].copy()
        root_rot=quaternion_to_rotation_6d(temp[3:7].copy())
        transl_jnts=temp[_C.INDEXING.TRANSL_JNT_IDXS].copy()
        rot_jnts=temp[_C.INDEXING.ROT_JNT_IDX].copy().reshape((-1, 3))
        converted_rot_jnts=axis_angle_to_rotation_6d(rot_jnts).flatten()

        q=jp.zeros(root_transl.shape[0]+root_rot.shape[0]+converted_rot_jnts.shape[0]+23)
        q=q.at[0:3].set(root_transl)
        q=q.at[3:9].set(root_rot)
        q=q.at[_C.INDEXING.R6D_TRANSL_IDXS].set(transl_jnts)
        q=q.at[_C.INDEXING.R6D_ROT_IDXS].set(converted_rot_jnts)

        return jp.concatenate([q, pipeline_state.qd])

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self._use_6d_notation:
            obs=self._convert_obs(pipeline_state=pipeline_state)
        else: 
            obs=jp.concatenate([pipeline_state.q, pipeline_state.qd])
        return obs
    
class SMPLHumanoid_scaling(PipelineEnv):

    # TODO:
    #   - fix issues with scaling
    #   - fix issues with model collapsing
    #   - tune PID controller values for position actuators
    #   - fix setting each position to initial value in step

    def __init__(self, use_newton_solver: Optional[bool]=True, use_6d_notation: Optional[bool]=False, ignore_joint_positions: Optional[bool]=True, **kwargs):
        path = 'lib/model/smpl_humanoid_v6.xml'
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

        self.initial_geoms=sys.geom_size
        self.initial_qpos=sys.init_q
        self.initial_body_pos=sys.body_pos # outdated, use geom_pos instead
        self.initial_mass=sys.body_mass
        self.initial_joint_lim=sys.jnt_range

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos=self.sys.init_q
        qvel=jp.zeros(self.sys.qd_size(),)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics={'counter': zero,}
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
    
    def _convert_obs(self, pipeline_state: base.State,) -> jax.Array:
        temp=pipeline_state.q.copy()
        root_transl=temp[0:3].copy()
        root_rot=quaternion_to_rotation_6d(temp[3:7].copy())
        transl_jnts=temp[_C.INDEXING.TRANSL_JNT_IDXS].copy()
        rot_jnts=temp[_C.INDEXING.ROT_JNT_IDX].copy().reshape((-1, 3))
        converted_rot_jnts=axis_angle_to_rotation_6d(rot_jnts).flatten()

        q=jp.zeros(root_transl.shape[0]+root_rot.shape[0]+converted_rot_jnts.shape[0]+23)
        q=q.at[0:3].set(root_transl)
        q=q.at[3:9].set(root_rot)
        q=q.at[_C.INDEXING.R6D_TRANSL_IDXS].set(transl_jnts)
        q=q.at[_C.INDEXING.R6D_ROT_IDXS].set(converted_rot_jnts)

        return jp.concatenate([q, pipeline_state.qd])

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self._use_6d_notation:
            obs=self._convert_obs(pipeline_state=pipeline_state)
        else: 
            obs=jp.concatenate([pipeline_state.q, pipeline_state.qd])
        return obs
    
if __name__=='__main__':
    env=SMPLHumanoid_basic()
    from lib.utils.viz import display_init_positions

    display_init_positions(env, False)