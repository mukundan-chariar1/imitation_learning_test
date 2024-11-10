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

        super().__init__(sys=sys, backend='generalized', **kwargs)

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = self.sys.init_q
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        return State(pipeline_state, obs, reward, done)
    
    def step(self, state: State, action: jax.Array) -> State:
        state_prev = state.pipeline_state
        new_state=self.pipeline_step(state_prev, action)
        reward=new_state.x.pos[0, 2]
        # print(reward) 
        # dict_keys(['q', 'qd', 'x', 'xd', 'contact', 'root_com', 'cinr', 'cd', 'cdof', 'cdofd', 'mass_mx', 'mass_mx_inv', 'con_jac', 'con_diag', 'con_aref', 'qf_smooth', 'qf_constraint', 'qdd'])
            # x dict_keys(['pos', 'rot'])
            # cinr dict_keys(['transform', 'i', 'mass'])
                # transform dict_keys(['pos', 'rot'])
            # cd dict_keys(['ang', 'vel'])
            # cdof dict_keys(['ang', 'vel'])
            # cdofd dict_keys(['ang', 'vel'])

        jst()
        
        obs = self._get_obs(new_state, action)
        done, zero = jp.zeros(2)

        return state.replace(
            pipeline_state=new_state, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        position = pipeline_state.q
        velocity = pipeline_state.qd

        return jp.concatenate([position, velocity])