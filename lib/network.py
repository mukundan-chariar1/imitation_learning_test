import torch
import torch.nn as nn
import torch.optim as optim

import jax
from jax import numpy as jp

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # First hidden layer with ReLU activation
        x = torch.tanh(self.fc2(x))  # Second hidden layer with ReLU activation
        x = torch.tanh(self.fc3(x))  # Output layer, also tanh for actions in [-1, 1] range
        return x

    def select_action(self, observation):
        """Use the MLP to select an action given the current observation."""
        with torch.no_grad():  # No need to compute gradients for action selection
            action = self(observation)
        return action  # Return action as a NumPy array for the environment

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueFunction, self).__init__()
        # Define a simple MLP with two hidden layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output a single value

    def forward(self, state):
        # Forward pass through the network
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)  # Output a single value for the given state
        return value
    
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



