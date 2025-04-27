from configs import constants as _C
from typing import Optional

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import jax
from jax import numpy as jp
from flax import nnx

import mujoco

from lib.environments.utils import *

from lib.irl.gail import *
from lib.irl.utils import *

from jax.debug import breakpoint as jst
from pdb import set_trace as st

class customHumanoid(PipelineEnv):
    """
    uses mujoco default humanoid model, modified for this application
    makes it jump
    """

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
    

class adversarialHumanoid(PipelineEnv):
    """
    uses mujoco default humanoid model, modified for this application
    uses a learned reward function to train it
    """

    def __init__(self, rng, disc_path, **kwargs):

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

        self.discriminator_config=load_config(disc_path.replace('discriminator', 'discriminator_config'))
        self.discriminator=load_model(Discriminator, rng, disc_path, **self.discriminator_config)
        self.discriminator.eval()
        self.relu=nnx.relu

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
        obs = self._get_obs(new_state, action)

        reward=self.relu(self.discriminator(jp.concatenate((state.obs, obs))))[0]
        # reward=self.discriminator(jp.concatenate((state.obs, obs)))[0]

        done=jp.where(new_state.x.pos[0, 2] >= jp.ones(1)*0.5, jp.zeros(1), jp.ones(1))[0]

        state.metrics.update(counter=count)
        return state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        position = pipeline_state.q
        velocity = pipeline_state.qd

        return jp.concatenate([position, velocity])