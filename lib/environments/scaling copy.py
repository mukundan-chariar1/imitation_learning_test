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
    joint_positions=env.initial_joints.copy()
    masses=env.initial_mass.copy()
    limits_min=sys.dof.limit[0].copy()
    limits_max=sys.dof.limit[1].copy()

    def randomize_heights(rng: jax.Array, qpos: jax.Array, capsule_shapes: jax.Array, joint_positions: jax.Array, masses: jax.Array, limits_min: jax.Array, limits_max: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        root_height=0

        for joint_idx, geom_idx in zip(_C.INDEXING.UNILATERAL_JNT_IDX, _C.INDEXING.UNILATERAL_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            val=(scaling_val-1)*joint_positions[geom_idx, 2]

            qpos=qpos.at[joint_idx].set(val)

            limits_min=limits_min.at[joint_idx-1].set(val*(1-_C.INDEXING.TRANS_JNT_TOL))
            limits_max=limits_max.at[joint_idx-1].set(val*(1+_C.INDEXING.TRANS_JNT_TOL))

            capsule_shapes=capsule_shapes.at[geom_idx].multiply(scaling_val)

            masses=masses.at[geom_idx].multiply(scaling_val)

        for joint_idx, geom_idx in zip(_C.INDEXING.LEG_JNT_IDX, _C.INDEXING.LEG_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            val=(scaling_val-1)*joint_positions[geom_idx, 2]

            qpos=qpos.at[joint_idx].set(-val)

            limits_min=limits_min.at[joint_idx-1].set(val*(1-_C.INDEXING.TRANS_JNT_TOL))
            limits_max=limits_max.at[joint_idx-1].set(val*(1+_C.INDEXING.TRANS_JNT_TOL))

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

            limits_min=limits_min.at[joint_idx1-1].set(val*(1-_C.INDEXING.TRANS_JNT_TOL))
            limits_max=limits_max.at[joint_idx1-1].set(val*(1+_C.INDEXING.TRANS_JNT_TOL))

            val=(scaling_val-1)*joint_positions[geom_idx2, 1]

            qpos=qpos.at[joint_idx2].set(-jp.abs(val))

            limits_min=limits_min.at[joint_idx2-1].set(-jp.abs(val*(1-_C.INDEXING.TRANS_JNT_TOL)))
            limits_max=limits_max.at[joint_idx2-1].set(-jp.abs(val*(1+_C.INDEXING.TRANS_JNT_TOL)))

            capsule_shapes=capsule_shapes.at[geom_idx1].multiply(scaling_val)
            capsule_shapes=capsule_shapes.at[geom_idx2].multiply(scaling_val)

            masses=masses.at[geom_idx1].multiply(scaling_val)
            masses=masses.at[geom_idx2].multiply(scaling_val)

        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
        val=(scaling_val-1)*joint_positions[_C.INDEXING.FOOT_GEOM_IDX[1], 0]

        qpos=qpos.at[_C.INDEXING.FOOT_JNT_IDX].set(val)

        limits_min=limits_min.at[_C.INDEXING.FOOT_JNT_IDX-1].set(val*(1-_C.INDEXING.TRANS_JNT_TOL))
        limits_max=limits_max.at[_C.INDEXING.FOOT_JNT_IDX-1].set(val*(1+_C.INDEXING.TRANS_JNT_TOL))
        
        capsule_shapes=capsule_shapes.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val)        

        qpos=qpos.at[_C.INDEXING.ROOT_JNT_IDX].add(root_height)
        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
        capsule_shapes=capsule_shapes.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val)
        
        return qpos, capsule_shapes, masses, limits_min, limits_max
    
    randomization_function=jax.jit(functools.partial(randomize_heights, qpos=qpos, capsule_shapes=capsule_shapes, joint_positions=joint_positions, masses=masses, limits_min=limits_min, limits_max=limits_max))

    qpos, capsule_shapes, masses, limits_min, limits_max=randomization_function(rng=rng)
    
    sys = sys.tree_replace({
        'init_q': qpos,
        'geom_size': capsule_shapes,
        'body_mass': masses,
        "dof.limit": (limits_min, limits_max)
    })

    jst()

    # sys = sys.replace(dof=sys.dof.replace(limit=(limits_min, limits_max)))


    return sys

def domain_randomize_no_vmap(sys: System, rng: jax.Array, env: Env) -> System:

    # TODO
    #   - fix randomization per reset due to tracer error
    #   - set joint limits for translation accordingly

    qpos=env.initial_qpos.copy()
    capsule_shapes=env.initial_geoms.copy()
    body_pos=env.initial_body_pos.copy()
    masses=env.initial_mass.copy()
    joint_limits=env.initial_joint_lim.copy()

    k=jp.where(joint_limits[:, 1]==1)

    jst()

    def randomize_heights(rng: jax.Array, qpos: jax.Array, capsule_shapes: jax.Array, body_pos: jax.Array, masses: jax.Array, joint_limits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        root_height=0

        for joint_idx, geom_idx in zip(_C.INDEXING.UNILATERAL_JNT_IDX, _C.INDEXING.UNILATERAL_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            val=(scaling_val-1)*body_pos[geom_idx, 2]

            qpos=qpos.at[joint_idx].set(val)
            joint_limits=joint_limits.at[joint_idx-6].set([val, val])

            capsule_shapes=capsule_shapes.at[geom_idx].multiply(scaling_val)
            masses=masses.at[geom_idx].multiply(scaling_val)

        for joint_idx, geom_idx in zip(_C.INDEXING.LEG_JNT_IDX, _C.INDEXING.LEG_GEOM_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            val=(scaling_val-1)*body_pos[geom_idx, 2]

            qpos=qpos.at[joint_idx].set(-val)
            joint_limits=joint_limits.at[joint_idx-6].set([-val, -val])

            capsule_shapes=capsule_shapes.at[geom_idx].multiply(scaling_val)
            masses=masses.at[geom_idx].multiply(scaling_val)

            root_height+=val.mean()
        
        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
        val=(scaling_val-1)*body_pos[_C.INDEXING.FOOT_GEOM_IDX[0], 2]

        qpos=qpos.at[joint_idx].set(val)
        joint_limits=joint_limits.at[_C.INDEXING.LEG_JNT_IDX[-1]-6].set([val, val])
        
        capsule_shapes=capsule_shapes.at[_C.INDEXING.FOOT_GEOM_IDX[0]].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.FOOT_GEOM_IDX[0]].multiply(scaling_val)

        for (joint_idx1, joint_idx2), (geom_idx1, geom_idx2) in zip(_C.INDEXING.BILATERAL_JNT_IDX, _C.INDEXING.BILATERAL_JNT_IDX):
            rng, subrng=jax.random.split(rng, 2)
            scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
            
            val=(scaling_val-1)*body_pos[geom_idx1, 1]
            qpos=qpos.at[joint_idx1].set(jp.abs(val))
            joint_limits=joint_limits.at[joint_idx1-6].set([jp.abs(val), jp.abs(val)])

            capsule_shapes=capsule_shapes.at[geom_idx1].multiply(scaling_val)
            masses=masses.at[geom_idx1].multiply(scaling_val)

            val=(scaling_val-1)*body_pos[geom_idx2, 1]
            qpos=qpos.at[joint_idx2].set(-jp.abs(val))
            joint_limits=joint_limits.at[joint_idx2-6].set([-jp.abs(val), -jp.abs(val)])

            capsule_shapes=capsule_shapes.at[geom_idx2].multiply(scaling_val)
            masses=masses.at[geom_idx2].multiply(scaling_val)
            

        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
        val=(scaling_val-1)*body_pos[_C.INDEXING.FOOT_GEOM_IDX[1], 0]

        qpos=qpos.at[_C.INDEXING.FOOT_JNT_IDX].set(val)

        joint_limits=joint_limits.at[_C.INDEXING.FOOT_JNT_IDX-6].set([val, val])
        
        capsule_shapes=capsule_shapes.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.FOOT_GEOM_IDX[1]].multiply(scaling_val)        

        qpos=qpos.at[_C.INDEXING.ROOT_JNT_IDX].add(root_height)
        rng, subrng=jax.random.split(rng, 2)
        scaling_val=jax.random.uniform(subrng, (), minval=0.75, maxval=1.25)
        capsule_shapes=capsule_shapes.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val)
        masses=masses.at[_C.INDEXING.ROOT_GEOM_IDX].multiply(scaling_val)
        
        return qpos, capsule_shapes, masses, joint_limits
    
    randomization_function=jax.jit(functools.partial(randomize_heights, qpos=qpos, capsule_shapes=capsule_shapes, body_pos=body_pos, masses=masses, joint_limits=joint_limits))

    qpos, capsule_shapes, masses, joint_limits=randomization_function(rng=rng)

    k=jp.where(joint_limits[:, 1]==1)
    
    sys = sys.tree_replace({
        'init_q': qpos,
        'geom_size': capsule_shapes,
        'body_mass': masses,
        "jnt_range": joint_limits,
    })

    jst()
    return sys