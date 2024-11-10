import cv2
import imageio
import numpy as np
import torch
import brax
from brax.envs import wrappers

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
        obs, reward, done, _ = env.step(action)

        # Render the frame, convert to BGR (for cv2), and add to frames list
        frame = env.render(mode='rgb_array')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        frames.append(frame)

        if done:
            break

    # Write frames to a video file using imageio
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved to {video_path}")

    return video_path