import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F

import jax
from jax import numpy as jp

from jax.debug import breakpoint as jst
from pdb import set_trace as st

# TODO:
# Change to jax based instead of tiorch to speed up things

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(ActorCritic, self).__init__()

        self.actor_layers=[nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(num_layers):
            self.actor_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.actor_layers.extend([nn.Linear(hidden_size, output_size), nn.ReLU(), torch.nn.Softmax(-1)])

        self.actor_layers=nn.Sequential(*self.actor_layers)

        self.critic_layers=[nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(num_layers):
            self.critic_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.critic_layers.append(nn.Linear(hidden_size, 1))

        self.critic_layers=nn.Sequential(*self.critic_layers)

    def select_action(self, obs):
        action_probs=self.actor_layers(obs)
        dist=Categorical(action_probs)
        action=dist.sample()
        log_probs=dist.log_prob(action)

        return action, log_probs
    
    def estimate_value(self, obs):
        value=self.critic_layers(obs)

        return value



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(MLP, self).__init__()
        self.layers=[nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(num_layers):
            self.layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.layers.extend([nn.Linear(hidden_size, output_size*2), nn.ReLU(), torch.nn.Softmax(-1)])

        self.layers=nn.Sequential(*self.layers)

    def forward(self, x):
        x=self.layers(x)
        logits = self.relu(x)  # Output layer, also tanh for actions in [-1, 1] range
        probs = self.softmax(x)
        return logits, probs

    def select_action(self, observation, env, epsilon=0.1):
        """Use the MLP to select an action given the current observation."""
        logits, action_probs = self(observation)
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + .001
        action_dist=Normal(loc, scale)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        action=torch.tanh(action)
            
        return action, log_prob  # Return action as a NumPy array for the environment
    
    def load_weights(self, path):
        """Loads weights from a specified path into the network."""
        try:
            self.load_state_dict(torch.load(path))
            print(f"Weights loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, num_layers=3):
        super(ValueFunction, self).__init__()
        # Define a simple MLP with two hidden layers
        self.layers=[nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for i in range(num_layers):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.layers.append(nn.Linear(hidden_dim, 1))  # Output a single value

        self.layers=nn.Sequential(*self.layers)

    def forward(self, state):
        # Forward pass through the network
        value = self.layers(state)  # Output a single value for the given state
        return value
    
    def load_weights(self, path):
        """Loads weights from a specified path into the network."""
        try:
            self.load_state_dict(torch.load(path))
            print(f"Weights loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    
def train_actor_critic(env, actor, critic, actor_optimizer, critic_optimizer, gamma=0.99, num_episodes=500):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Actor chooses action based on policy
            action_probs = actor(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()  # Sample action according to action probabilities
            log_prob = action_dist.log_prob(action)  # Log probability of the chosen action

            # Take action in the environment
            next_state, reward, done, _ = env.step(action.item())
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)

            # Calculate value of current state and next state
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            
            # Calculate TD error and Critic loss
            target = reward_tensor + (1 - done) * gamma * next_value
            td_error = target - value
            critic_loss = td_error.pow(2).mean()  # Mean Squared TD Error

            # Critic update
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor loss and update
            actor_loss = -log_prob * td_error.detach()  # Maximize expected reward
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Move to next state
            state = next_state
            episode_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")



