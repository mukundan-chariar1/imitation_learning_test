from configs import constants as _C

import functools
from typing import Optional

from brax.envs.base import Env
from brax import System

import jax
from jax import numpy as jp

import mujoco

from lib.environments.utils import *

from jax.debug import breakpoint as jst
from pdb import set_trace as st

# import torch

def domain_randomize(sys: System, rng: jax.Array, env: Env) -> tuple[System, System]:

    # TODO
    #   - fix randomization per reset due to tracer error
    #   - set joint limits for translation accordingly

    qpos=env.initial_qpos.copy()
    capsule_shapes=env.initial_geoms.copy()
    joint_positions=env.initial_joints.copy()
    masses=env.initial_mass.copy()

    def randomize_heights(rng: jax.Array, qpos: jax.Array, capsule_shapes: jax.Array, joint_positions: jax.Array, masses: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        root_height=0

        for joint_idx, geom_idx in zip(_C.INDEXING.UNILATERAL_JNT_IDX, _C.INDEXING.UNILATERAL_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            val=(scaling_val-1)*joint_positions[geom_idx, 2]
            qpos=qpos.at[joint_idx].set(val)
            capsule_shapes=capsule_shapes.at[geom_idx].multiply(scaling_val)
            masses=masses.at[geom_idx].multiply(scaling_val)

        for joint_idx, geom_idx in zip(_C.INDEXING.LEG_JNT_IDX, _C.INDEXING.LEG_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            val=(scaling_val-1)*joint_positions[geom_idx, 2]
            qpos=qpos.at[joint_idx].set(-val)
            if geom_idx is not None: 
                capsule_shapes=capsule_shapes.at[geom_idx].multiply(scaling_val)
                masses=masses.at[geom_idx].multiply(scaling_val)
            root_height+=val.mean() # hack, needs to be fixed
        
        capsule_shapes=capsule_shapes.at[_C.INDEXING.FOOT_GEOM_IDX[0]].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.FOOT_GEOM_IDX[0]].multiply(scaling_val)

        for (joint_idx1, joint_idx2), (geom_idx1, geom_idx2) in zip(_C.INDEXING.BILATERAL_JNT_IDX, _C.INDEXING.BILATERAL_JNT_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            val=(scaling_val-1)*joint_positions[geom_idx1, 1]
            qpos=qpos.at[joint_idx1].set(jp.abs(val))
            val=(scaling_val-1)*joint_positions[geom_idx2, 1]
            qpos=qpos.at[joint_idx2].set(-jp.abs(val))
            capsule_shapes=capsule_shapes.at[geom_idx1].multiply(scaling_val)
            capsule_shapes=capsule_shapes.at[geom_idx2].multiply(scaling_val)
            masses=masses.at[geom_idx1].multiply(scaling_val)
            masses=masses.at[geom_idx2].multiply(scaling_val)

        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
        val=(scaling_val-1)*joint_positions[_C.INDEXING.FOOT_GEOM_IDX[1], 0]
        qpos=qpos.at[_C.INDEXING.FOOT_JNT_IDX].set(val)
        capsule_shapes=capsule_shapes.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val)        

        qpos=qpos.at[_C.INDEXING.ROOT_JNT_IDX].add(root_height)
        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
        capsule_shapes=capsule_shapes.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val)
        
        return qpos, capsule_shapes, masses
    
    randomization_function=jax.vmap(functools.partial(randomize_heights, qpos=qpos, capsule_shapes=capsule_shapes, joint_positions=joint_positions, masses=masses))

    qpos, capsule_shapes, masses=randomization_function(rng=rng)
        
    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'init_q': 0,
        'geom_size': 0,
        'body_mass': 0,
    })

    sys = sys.tree_replace({
        'init_q': qpos,
        'geom_size': capsule_shapes,
        'body_mass': masses,
    })

    return sys, in_axes

def domain_randomize_no_vmap(sys: System, rng: jax.Array, env: Env) -> System:

    # TODO
    #   - fix randomization per reset due to tracer error
    #   - set joint limits for translation accordingly

    qpos=env.initial_qpos.copy()
    capsule_shapes=env.initial_geoms.copy()
    body_pos=env.initial_body_pos.copy()
    masses=env.initial_mass.copy()
    joint_limits=env.initial_joint_lim.copy()

    def randomize_heights(rng: jax.Array, qpos: jax.Array, capsule_shapes: jax.Array, body_pos: jax.Array, masses: jax.Array, joint_limits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        root_height=0

        for joint_idx, geom_idx in zip(_C.INDEXING.UNILATERAL_JNT_IDX, _C.INDEXING.UNILATERAL_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.5)
            val=(scaling_val-1)*body_pos[geom_idx, 2]

            qpos=qpos.at[joint_idx].set(val)
            joint_limits=joint_limits.at[joint_idx-6].set([val, val])

            capsule_shapes=capsule_shapes.at[geom_idx].multiply(scaling_val*0.75)
            # masses=masses.at[geom_idx].multiply(scaling_val*0.75)

        for joint_idx, geom_idx in zip(_C.INDEXING.LEG_JNT_IDX, _C.INDEXING.LEG_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.5)
            val=(scaling_val-1)*body_pos[geom_idx, 2]

            qpos=qpos.at[joint_idx].set(-val)
            joint_limits=joint_limits.at[joint_idx-6].set([-val, -val])

            capsule_shapes=capsule_shapes.at[geom_idx].multiply(scaling_val*0.75)
            # masses=masses.at[geom_idx].multiply(scaling_val*0.75)

            root_height+=val.mean()
        
        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.5)
        val=(scaling_val-1)*body_pos[_C.INDEXING.FOOT_GEOM_IDX[0], 2]

        qpos=qpos.at[_C.INDEXING.FOOT_JNT_IDX[0]].set(-val)
        joint_limits=joint_limits.at[_C.INDEXING.FOOT_JNT_IDX[0]-6].set([-val, -val])
        
        capsule_shapes=capsule_shapes.at[_C.INDEXING.FOOT_GEOM_IDX[0]].multiply(2)
        # masses=masses.at[_C.INDEXING.FOOT_GEOM_IDX[0]].multiply(scaling_val*0.75)

        root_height+=val.mean()

        for (joint_idx1, joint_idx2), (geom_idx1, geom_idx2) in zip(_C.INDEXING.BILATERAL_JNT_IDX, _C.INDEXING.BILATERAL_JNT_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.5)
            
            val=(scaling_val-1)*body_pos[geom_idx1, 1]
            qpos=qpos.at[joint_idx1].set(val)
            joint_limits=joint_limits.at[joint_idx1-6].set([val, val])

            capsule_shapes=capsule_shapes.at[geom_idx1].multiply(scaling_val*0.75)
            # masses=masses.at[geom_idx1].multiply(scaling_val*0.75)

            val=(scaling_val-1)*body_pos[geom_idx2, 1]
            qpos=qpos.at[joint_idx2].set(-val)
            joint_limits=joint_limits.at[joint_idx2-6].set([-val, -val])

            capsule_shapes=capsule_shapes.at[geom_idx2].multiply(scaling_val*0.75)
            # masses=masses.at[geom_idx2].multiply(scaling_val*0.75)

        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.5)
        val=(scaling_val-1)*body_pos[_C.INDEXING.FOOT_GEOM_IDX[1], 0]

        qpos=qpos.at[_C.INDEXING.FOOT_JNT_IDX[1]].set(val)

        joint_limits=joint_limits.at[_C.INDEXING.FOOT_JNT_IDX[1]-6].set([val, val])
        
        capsule_shapes=capsule_shapes.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val*0.75)
        # masses=masses.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val*0.75)        

        qpos=qpos.at[_C.INDEXING.ROOT_JNT_IDX].add(root_height)
        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.5)
        capsule_shapes=capsule_shapes.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val*0.75)
        # masses=masses.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val*0.75)
        
        return qpos, capsule_shapes, masses, joint_limits
    
    randomization_function=jax.jit(functools.partial(randomize_heights, qpos=qpos, capsule_shapes=capsule_shapes, body_pos=body_pos, masses=masses, joint_limits=joint_limits))

    qpos, capsule_shapes, masses, joint_limits=randomization_function(rng=rng)
    
    sys = sys.tree_replace({
        'init_q': qpos,
        'geom_size': capsule_shapes,
        # 'body_mass': masses,
        "jnt_range": joint_limits,
    })

    return sys

def domain_randomize_no_vmap_temp(sys: System, rng: jax.Array, env: Env) -> System:

    qpos=sys.init_q.copy()
    limits_min=sys.dof.limit[0].copy()
    limits_max=sys.dof.limit[1].copy()
    # limits_bool=sys.jnt_limited.copy()
    sys_jnt_limits=sys.jnt_range.copy()

    jst()

    def randomize_heights(rng, qpos, sys_jnt_limits):
        height = jax.random.uniform(rng, shape=())
        qpos=qpos.at[-1].set(height)
        sys_jnt_limits=sys_jnt_limits.at[-1, 0].set(height)
        sys_jnt_limits=sys_jnt_limits.at[-1, 1].set(height)
        return qpos, sys_jnt_limits
    
    randomization_function=jax.jit(functools.partial(randomize_heights, qpos=qpos, sys_jnt_limits=sys_jnt_limits))

    qpos, sys_jnt_limits=randomization_function(rng=rng)
    
    sys = sys.tree_replace({
        "init_q": qpos,
        # "dof.limit": (limits_min, limits_max),
        # "jnt_limited": limits_bool
        "jnt_range": sys_jnt_limits,
    })

    # sys.jnt_limits=limits_bool

    # jst()

    # sys = sys.replace(dof=sys.dof.replace(limit=(limits_min, limits_max)))

    # jst()


    return sys