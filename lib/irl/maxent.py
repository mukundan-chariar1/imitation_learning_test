import jax
import jax.numpy as jnp
import brax
from brax import envs
from brax.io import html
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
import numpy as np
from typing import Any, Tuple, Optional, Dict, List
from flax import struct

from lib.utils.trajectory import get_observation

# First, let's define some helper functions and types

@struct.dataclass
class Trajectory:
    observations: jnp.ndarray  # [T, obs_dim]
    actions: jnp.ndarray  # [T, act_dim]
    rewards: jnp.ndarray  # [T]
    next_observations: jnp.ndarray  # [T, obs_dim]
    dones: jnp.ndarray  # [T]

def collect_trajectories(
    env_name: str,
    num_trajectories: int = 50,
    trajectory_length: int = 1000,
    policy: Optional[Any] = None,
    random_policy: bool = True,
    params: Optional[Any] = None,
) -> List[Trajectory]:
    """Collect trajectories from the environment."""
    env = envs.create(env_name)
    if random_policy:
        # Use random policy if no policy provided
        def random_action(rng, obs):
            return jax.random.uniform(rng, (env.action_size,), minval=-1, maxval=1)
    else:
        # Use provided policy
        def random_action(rng, obs):
            return policy(obs, params)
    
    trajectories = []
    
    for _ in range(num_trajectories):
        rng = jax.random.PRNGKey(np.random.randint(0, 10000))
        state = env.reset(rng)
        obs = []
        acts = []
        rews = []
        next_obs = []
        dones = []
        
        for __ in range(trajectory_length):
            rng, act_rng = jax.random.split(rng)
            action = random_action(act_rng, state.obs)
            next_state = env.step(state, action)
            
            obs.append(state.obs)
            acts.append(action)
            rews.append(next_state.reward)
            next_obs.append(next_state.obs)
            dones.append(next_state.done)
            
            state = next_state
            if next_state.done:
                break
        
        trajectory = Trajectory(
            observations=jnp.array(obs),
            actions=jnp.array(acts),
            rewards=jnp.array(rews),
            next_observations=jnp.array(next_obs),
            dones=jnp.array(dones)
        )
        trajectories.append(trajectory)
    
    return trajectories

def extract_features(trajectory: Trajectory) -> jnp.ndarray:
    """Extract features from a trajectory (simple state features)."""
    # This is a simple example - you might want more sophisticated features
    return trajectory.observations

def maxent_irl(trajectories: List[Trajectory], 
               num_iterations: int = 100,
               learning_rate: float = 0.01) -> jnp.ndarray:
    """
    Maximum Entropy Inverse Reinforcement Learning to recover reward weights.
    
    Args:
        trajectories: List of demonstration trajectories
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Estimated reward weights
    """
    # Convert trajectories to features
    features = [extract_features(traj) for traj in trajectories]
    demo_features = jnp.concatenate(features, axis=0)
    
    # Initialize reward weights
    reward_weights = jax.random.normal(jax.random.PRNGKey(0), (demo_features.shape[1],))
    
    # Define the loss function
    def loss_fn(weights):
        # Calculate expert feature expectations
        expert_feature_exp = jnp.mean(demo_features, axis=0)
        
        # Calculate predicted feature expectations (using current policy)
        # For simplicity, we'll use the empirical feature expectations from the demos
        # In a full implementation, you'd need to learn a policy with current rewards
        # and sample trajectories from it
        predicted_feature_exp = expert_feature_exp  # Placeholder
        
        # Maximum entropy loss
        loss = jnp.sum((expert_feature_exp - predicted_feature_exp) ** 2)
        return loss
    
    # Optimize the weights
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(reward_weights)
    
    @jax.jit
    def update_step(weights, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state, loss
    
    for _ in range(num_iterations):
        reward_weights, opt_state, current_loss = update_step(reward_weights, opt_state)
        if _ % 10 == 0:
            print(f"Iteration {_}, Loss: {current_loss}")
    
    return reward_weights

def recovered_reward_function(obs: jnp.ndarray, reward_weights: jnp.ndarray) -> jnp.ndarray:
    """Compute reward using the recovered weights."""
    return jnp.dot(obs, reward_weights)

# Example usage
if __name__ == "__main__":
    # 1. Collect demonstration trajectories (expert or random)
    print("Collecting demonstration trajectories...")
    expert_trajectories = collect_trajectories("ant", num_trajectories=20, random_policy=False)
    # Note: In practice, you'd want real expert trajectories
    
    # 2. Learn reward weights using MaxEnt IRL
    print("\nLearning reward function...")
    reward_weights = maxent_irl(expert_trajectories)
    
    # 3. Define the recovered reward function
    def recovered_reward(obs):
        return recovered_reward_function(obs, reward_weights)
    
    print("\nRecovered reward weights:", reward_weights)
    print("Example reward for sample observation:", 
          recovered_reward(jnp.zeros_like(reward_weights)))