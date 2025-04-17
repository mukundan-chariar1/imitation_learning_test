from jax import numpy as jp
import jax

from jax.debug import breakpoint as jst
from pdb import set_trace as st

# import torch

def quaternion_to_matrix(quaternions: jax.Array) -> jax.Array:
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_rotation_6d(matrix: jax.Array) -> jax.Array:
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def quaternion_to_rotation_6d(quaternion: jax.Array) -> jax.Array:
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))

def axis_angle_to_quaternion(axis_angle: jax.Array) -> jax.Array:
    # Compute the angle (magnitude of the vector)
    angles = jp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-8

    sin_half_angles_over_angles=jp.where(jp.abs(angles)<eps, 0.5-(angles**2)/48, jp.sin(half_angles)/angles)

    # Compute the quaternion
    quaternions = jp.concatenate(
        [jp.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )

    quaternions=quaternions / (jp.linalg.norm(quaternions, axis=-1, keepdims=True) + eps)

    return quaternions

def axis_angle_to_matrix(axis_angle: jax.Array) -> jax.Array:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_rotation_6d(axis_angle: jax.Array):
    return quaternion_to_rotation_6d(axis_angle_to_quaternion(axis_angle))

def quaternion_to_axis_angle(quaternion: jp.ndarray) -> jp.ndarray:
    """
    Convert a quaternion to axis-angle representation.

    Args:
        quaternion: A quaternion array of shape (..., 4).

    Returns:
        axis_angle: An axis-angle array of shape (..., 3).
    """
    # Normalize the quaternion to ensure it's a unit quaternion
    quaternion = quaternion / (jp.linalg.norm(quaternion, axis=-1, keepdims=True)+1e-8)

    # Extract the real part (scalar) and imaginary part (vector)
    w = quaternion[..., 0]  # Scalar part
    xyz = quaternion[..., 1:]  # Vector part

    # Compute the angle
    angle = 2 * jp.arccos(jp.clip(w, -1.0, 1.0))  # Clip to avoid numerical instability

    # Compute the axis
    norm_xyz = jp.linalg.norm(xyz, axis=-1, keepdims=True)
    axis = jp.where(norm_xyz > 1e-8, xyz / norm_xyz, jp.array([1.0, 0.0, 0.0]))

    # Combine axis and angle into axis-angle representation
    axis_angle = axis * angle[..., None]

    return axis_angle