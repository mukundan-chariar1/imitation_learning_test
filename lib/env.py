from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco
from jax.debug import breakpoint as jst
from pdb import set_trace as st

class customHumanoid(PipelineEnv):
    def __init__(self, **kwargs):

        path = epath.resource_path('brax') / 'envs/assets/humanoid.xml'
        sys=mjcf.load(path)

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
        vel_reward=(new_state.root_com[0, :]-state_prev.root_com[0, :])/self.dt
        vel_reward=jp.where(vel_reward>=jp.ones(3)*0.5, vel_reward*self._vel_reward_weight, 0)[-1]

        reward=vel_reward+upward_reward+self._healthy_reward

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
    def __init__(self, **kwargs):
        path = 'lib/model/smpl_humanoid_v2.xml'
        sys=mjcf.load(path)

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(sys=sys, **kwargs)

        self._healthy_reward=5.0
        self._upward_reward_weight=5
        self._vel_reward_weight=15

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = self.sys.init_q
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
        upward_reward=jp.where(new_state.x.pos[9, :] >= jp.ones(3)*0.5, new_state.x.pos[9, :]**2*self._upward_reward_weight, 0)[-1]
        vel_reward=(new_state.root_com[9, :]-state_prev.root_com[9, :])/self.dt
        vel_reward=jp.where(vel_reward>=jp.ones(3)*0.5, vel_reward*self._vel_reward_weight, 0)[-1]

        reward=vel_reward+upward_reward

        obs = self._get_obs(new_state, action)

        done=jp.where(new_state.x.pos[9, 2] >= jp.ones(1)*0.5, jp.zeros(1), jp.ones(1))[0]

        # if count<=2 and done: jst()

        state.metrics.update(counter=count)
        return state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        position = pipeline_state.q
        velocity = pipeline_state.qd

        return jp.concatenate([position, velocity])