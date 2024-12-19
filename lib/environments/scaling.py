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

def domain_randomize_for_viz(sys, rng):
    """Applies domain randomization to a Brax system."""
    

    # Randomize link lengths
    rng, key_lengths = jax.random.split(rng)
    lengths = jax.random.uniform(key_lengths, (sys.geom_size.shape[0],), minval=0.5*sys.geom_size[:, 0], maxval=1.5*sys.geom_size[:, 0])
    sys = sys.tree_replace({'geom_size': sys.geom_size.at[:, 0].set(lengths)})

    return sys

def domain_randomize(sys, rng):
    """Randomizes various properties of the Brax system, including link lengths."""

    #['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    
    # @jax.vmap
    def rand(rng):
        # Split RNG for multiple randomizations
        rng, key_friction, key_gain, key_bias, key_length = jax.random.split(rng, 5)
    
        # Randomize link lengths
        length_range = (0.5, 1.5)  # Scale link lengths between 50% and 150% of original
        lengths = jax.random.uniform(
            key_length, (sys.geom_size.shape[0],), minval=length_range[0], maxval=length_range[1]
        )
        geom_size = sys.geom_size.at[:, 0].set(lengths)  # Assume lengths are stored in the first dimension

        return geom_size

    # Apply randomization
    geom_size = rand(rng)
    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_size': 0,  # Add this line to specify the in_axes for 'geom_size
    })

    # Replace the randomized parameters in the system
    sys = sys.tree_replace({
        'geom_size': geom_size,
    })

    return sys, in_axes