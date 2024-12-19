from jax import numpy as jp

import torch

def quaternion_to_matrix(quaternions):
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


def matrix_to_rotation_6d(matrix):
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def quaternion_to_rotation_6d(quaternion):
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    axis_angle[:, 0]+=torch.pi
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_quaternion_np(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis-angle form,
                    as a NumPy array of shape (..., 3), where the magnitude
                    is the angle turned anticlockwise in radians around the
                    vector's direction.

    Returns:
        Quaternions with real part first, as a NumPy array of shape (..., 4).
    """
    # Compute the angle (magnitude of the vector)
    angles = jp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6

    # Identify small angles to handle them separately
    small_angles = jp.abs(angles) < eps

    # Allocate the sine term array
    sin_half_angles_over_angles = jp.empty_like(angles)

    # For large angles: sin(θ/2) / θ
    sin_half_angles_over_angles[~small_angles] = (
        jp.sin(half_angles[~small_angles]) / angles[~small_angles]
    )

    # For small angles: Approximation using Taylor expansion
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] ** 2) / 48
    )

    # Compute the quaternion
    quaternions = jp.concatenate(
        [jp.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    
    return quaternions