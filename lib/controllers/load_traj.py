import pickle
import os
import os.path as osp

import jax.numpy as jp
# from jax.scipy.interpolate import CubicSpline

import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.interpolate import CubicSpline

from pdb import set_trace as st
from jax.debug import breakpoint as jst

def upsample_signal_cubic(original_signal, original_dt=1/40, upsampled_dt=0.003*5):
    """
    Upsample a signal using cubic spline interpolation.

    Parameters:
        original_signal (numpy array): Values of the original signal.
        original_dt (float): Time step of the original signal.
        upsampled_dt (float): Desired time step for the upsampled signal.

    Returns:
        upsampled_time (numpy array): New time points after upsampling.
        upsampled_signal (numpy array): Upsampled signal values.
    """
    # Generate original time points
    original_time = np.arange(0, len(original_signal) * original_dt, original_dt)

    # Generate new time points for upsampling
    upsampled_time = np.arange(0, original_time[-1], upsampled_dt)

    # Perform cubic spline interpolation
    cubic_spline = CubicSpline(original_time, original_signal)
    upsampled_signal = cubic_spline(upsampled_time)

    return upsampled_signal

def convert_traj_to_pkl():
    import torch
    result=torch.load('/home/mukundan/Desktop/Summer_SEM/imitation_learning/dataset/data/rep_00_output.pt')#['full_pose'].reshape(-1, 24, 3)
    savefile={key: result[key].numpy() for key in ['pose_embedding', 'full_pose', 'transl', 'betas', 'side_R', 'side_T']}

    with open('/home/mukundan/Desktop/Summer_SEM/imitation_learning/dataset/data/data.pickle', 'wb') as file:
        pickle.dump(savefile, file)

def get_traj_from_pkl():
    with open('dataset/data/data.pickle', 'rb') as file:
        pose=pickle.load(file)['full_pose'].reshape((-1, 24, 3))

    rot=upsample_signal_cubic(pose.copy())
    ang=upsample_signal_cubic(pose.copy())
    ang=np.concatenate(((ang[1:]-ang[:-1])/40, np.zeros((1, 24, 3))), axis=0)
    # ang=np.concatenate((np.zeros((1, 24, 3)), (ang[1:]-ang[:-1])/40), axis=0)
    transl=np.zeros((ang.shape[0], 3), dtype=float)

    return rot, ang, transl, transl

def get_traj_from_wham():
    results=joblib.load('test_data/drone_video/wham_output_modified.pkl')

    pose=results[0]['smpl_pose'].reshape((-1, 24, 3))
    transl=results[0]['trans_world']
    # pose=results[0]['pose'].reshape((-1, 24, 3))
    # transl=results[0]['trans']

    # fps=int(results[0]['fps']) if results[0]['fps']==int(results[0]['fps']) else int(results[0]['fps'])+1
    fps=30

    rot=upsample_signal_cubic(pose.copy(), original_dt=1/fps)
    ang=upsample_signal_cubic(pose.copy(), original_dt=1/fps)
    ang=np.concatenate(((ang[1:]-ang[:-1])/fps, np.zeros((1, 24, 3))), axis=0)
    transl=upsample_signal_cubic(transl.copy(), original_dt=1/fps)
    transl=np.column_stack([transl[:, 0], transl[:, 2], np.full(transl.shape[0], transl[:, 1].mean())])
    # transl=np.column_stack([transl[:, 0], transl[:, 2], -transl[:, 1]])
    vel=np.concatenate(((transl[1:]-transl[:-1])/fps, np.zeros((1, 3))), axis=0)
    return rot[:, :, [1, 2, 0]], ang[:, :, [2, 0, 1]], transl, vel

def plot_3d_trajectory(points, title="3D Trajectory", xlabel="X", ylabel="Y", zlabel="Z", show=True, save_path=None):
    """
    Plot a 3D trajectory from a sequence of points.
    
    Parameters:
    - points: Array-like of shape (n_points, 3) containing (x,y,z) coordinates
    - title: Plot title (default: "3D Trajectory")
    - xlabel/ylabel/zlabel: Axis labels (default: "X"/"Y"/"Z")
    - show: Whether to display the plot (default: True)
    - save_path: If provided, saves the plot to this path (default: None)
    """
    # Convert to numpy array if not already
    points = np.asarray(points)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 
            marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.8)
    
    # Plot starting and ending points
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], 
               c='green', s=100, label='Start', marker='*')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
               c='red', s=100, label='End', marker='*')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Requires matplotlib >= 3.3.0
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig, ax

if __name__=='__main__':
    rot, ang, transl, vel=get_traj_from_wham()
    plot_3d_trajectory(transl)

    import pdb; pdb.set_trace()

