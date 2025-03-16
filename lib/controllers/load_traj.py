import pickle

import jax.numpy as jp
# from jax.scipy.interpolate import CubicSpline

import numpy as np
from scipy.interpolate import CubicSpline

def upsample_signal_cubic(original_signal, original_dt=1/40, upsampled_dt=0.003/5):
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

def get_traj():
    with open('/home/mukundan/Desktop/Summer_SEM/imitation_learning/dataset/data/data.pickle', 'rb') as file:
        pose=pickle.load(file)['full_pose'].reshape((-1, 24, 3))

    rot=upsample_signal_cubic(pose.copy())
    ang=upsample_signal_cubic(pose.copy())
    ang=np.concatenate(((ang[1:]-ang[:-1])/40, np.zeros((1, 24, 3))), axis=0)
    # ang=np.concatenate((np.zeros((1, 24, 3)), (ang[1:]-ang[:-1])/40), axis=0)

    return rot, ang

if __name__=='__main__':
    rot, ang=get_traj()