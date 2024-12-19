import jax
from jax import numpy as jp
from jax import nn
from jax import random

import flax
from flax import nnx

import optax

from lib.environments.utils import *

from jax.debug import breakpoint as jst
from pdb import set_trace as st

class MLP(nnx.Module):
    def __init__(self, input_size, hidden_size, output_size, key, num_layers=3, activation=nnx.relu):
        super(MLP, self).__init__()
        self.input_layer = nnx.Linear(input_size, hidden_size, rngs=key)
        # self.hidden_layers = [nnx.Linear(hidden_size, hidden_size, rngs=key) for _ in range(num_layers)]
        self.hidden_layers=[] if not activation else [activation]
        for i in range(num_layers): 
            self.hidden_layers.append(nnx.Linear(hidden_size, hidden_size, rngs=key))
            if activation: self.hidden_layers.append(activation)
        self.hidden_layers=nnx.Sequential(*self.hidden_layers)
        self.output_layer = nnx.Linear(hidden_size, output_size * 2, rngs=key)

    def __call__(self, x):
        out=self.output_layer(self.hidden_layers(self.input_layer(x)))
        # logits = nnx.tanh(x)  # Output layer, also for actions in [-1, 1] range
        # probs = nnx.softmax(x, axis=-1)
        return out
    
class ActorCritic(nnx.Module):
    def __init__(self, state_dim, action_dim, hidden_size, key, actor_layers=3, critic_layers=1):
        super(ActorCritic, self).__init__()

        # self.actor = nnx.Sequential([
        #     nnx.Linear(state_dim, hidden_size, rngs=key),
        #     nnx.relu,
        #     nnx.Linear(hidden_size, action_dim, rngs=key),
        #     nnx.softmax
        # ])

        # # Critic network for state-value estimation
        # self.critic = nnx.Sequential([
        #     nnx.Linear(state_dim, hidden_size, rngs=key),
        #     nnx.relu,
        #     nnx.Linear(hidden_size, 1, rngs=key)  # Single value output
        # ])

        self.actor=MLP(state_dim, hidden_size, action_dim, key, actor_layers)
        self.softmax=nnx.softmax

        self.critic=MLP(state_dim, hidden_size, 1, key, critic_layers)

    # def init(self, key):
    #     """Initialize parameters for both networks."""
        
    #     self.actor_params = self.actor.init(actor_key, jp.ones((1, self.state_dim)))
    #     self.critic_params = self.critic.init(critic_key, jp.ones((1, self.state_dim)))

    def forward_actor(self, state):
        return self.softmax(self.actor(state))

    def forward_critic(self, state):
        return self.critic(state)

@jax.jit
def update_model(state, action, reward, next_state, done, model, actor_opt_state, critic_opt_state):
    def actor_loss_fn(model, state, next_state, action):
        probs = model.forward_actor(state)
        log_prob = jp.log(probs[action])
        advantage = reward + gamma * (1 - done) * model.forward_critic(next_state) - model.forward_critic(state)
        return -log_prob * advantage

    def critic_loss_fn():
        value = model.forward_critic(state)
        target = reward + gamma * (1 - done) * model.forward_critic(next_state)
        return jp.mean((value - target) ** 2)

    # Update actor
    actor_grad_fn = nnx.value_and_grad(actor_loss_fn, has_aux=True)
    (actor_loss, action_logits), actor_grads = actor_grad_fn(model.actor, state)
    model.actor_params = optax.apply_updates(model.actor_params, actor_updates)

    # Update critic
    critic_grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)
    critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state)
    model.critic_params = optax.apply_updates(model.critic_params, critic_updates)

    return model, actor_opt_state, critic_opt_state

if __name__=='__main__':
    # Example usage
    # key = random.PRNGKey(0)
    # x = jp.array([[0.1, 0.2, 0.3]])  # Dummy input

    # # Define the model
    # model = MLP(input_size=3, hidden_size=128, output_size=10, num_layers=3, key=nnx.Rngs(0))

    # # Initialize parameters
    # # variables = model.init(key, x)

    # # Forward pass
    # logits, probs = model(x)
    # print("Logits:", logits)
    # print("Probabilities:", probs)

    state_dim = 4  # Example state dimension
    action_dim = 2  # Example action dimension
    hidden_size = 128
    learning_rate = 0.001
    gamma = 0.99

    # Initialize Actor-Critic model and optimizer
    # key = random.PRNGKey(0)
    key = nnx.Rngs(0)
    model = ActorCritic(state_dim, action_dim, hidden_size, key=key)
    # model.init(key)
    actor_optimizer = nnx.optimizer(model.actor, optax.adam(learning_rate))
    critic_optimizer = nnx.optimizer(model.critic, optax.adam(learning_rate))

    # Example usage
    state = jp.array([[0.1, 0.2, 0.3, 0.4]])  # Example state
    action = 0  # Example action
    reward = 1.0  # Example reward
    next_state = jp.array([[0.5, 0.6, 0.7, 0.8]])  # Example next state
    done = 0  # Not done

    model, actor_opt_state, critic_opt_state = update_model(state, action, reward, next_state, done, model, actor_opt_state, critic_opt_state)
