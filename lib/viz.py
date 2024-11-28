import cv2
import imageio
import numpy as np
import torch
import brax
from brax.envs import wrappers

import gym
import base64
from IPython.display import HTML
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from IPython.display import HTML, clear_output

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html

from jax.debug import breakpoint as jst
from pdb import set_trace as st

@torch.no_grad
def save_video_unroll(env, policy, num_steps=200, video_path="rollout.mp4"):
    """
    Runs one unroll of the environment using a policy and saves it as a video.
    
    Args:
        env: The environment instance wrapped with gym_wrapper.
        policy: A function that takes an observation and returns an action.
        num_steps: Number of steps to visualize.
        video_path: Path to save the output video.
    """
    # Initialize environment and list for storing frames
    obs = env.reset()
    frames = []

    # Run one unroll of the environment
    for _ in range(num_steps):
        # Use the policy to select an action
        action, log_probs = policy.select_action(obs, env)
        # action = action.detach().cpu().numpy()  # Convert action to numpy for gym compatibility

        # Take a step in the environment
        # for u in range(10):
        obs, rewards, done, _ = env.step(action)
        frame = env.render(mode='rgb_array')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        frames.append(frame)
        if done: break

        if done:
            break

    # Write frames to a video file using imageio
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved to {video_path}")

    return video_path

def create_interactive_rollout(env, policy, num_steps=200):

    rollout = []
    # pipeline_state, obs = env.reset()
    obs = env.reset()
    done=False
    while not done:
        # rollout.append(pipeline_state)
        action, log_probs = policy.select_action(obs, env)
        obs, rewards, done, _ = env.step(action)
        env.render()
        # pipeline_state, obs, rewards, done, _ = env.step(action)
        

    # display(HTML(html.render(env.sys_.tree_replace({'opt.timestep': env.dt_}), rollout)))
    # env.sys_.tree_replace({'opt.timestep': env.dt_}), rollout)
    # env.render()