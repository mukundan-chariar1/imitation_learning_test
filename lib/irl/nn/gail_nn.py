from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Optional
from jax import random

class Generator(nn.Module):
    latent_size: int
    output_size: int
    hidden_sizes: Sequence[int] = (128, 256, 256)
    batchnorm: bool = True
    activation: str = 'relu'
    
    def setup(self):
        act = getattr(nn, self.activation)
        
        # Encoder layers
        encoder_layers = []
        encoder_layers.append(nn.Dense(self.hidden_sizes[-1]))
        for in_size, out_size in reversed(list(zip(self.hidden_sizes[1:], self.hidden_sizes[:-1]))):
            encoder_layers.append(nn.Dense(out_size))
            if self.batchnorm:
                encoder_layers.append(nn.BatchNorm(use_running_average=False))
            encoder_layers.append(act)
        encoder_layers.append(nn.Dense(self.latent_size * 2))
        self.encoder = nn.Sequential(encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        decoder_layers.append(nn.Dense(self.hidden_sizes[0]))
        for in_size, out_size in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:]):
            decoder_layers.append(nn.Dense(out_size))
            if self.batchnorm:
                decoder_layers.append(nn.BatchNorm(use_running_average=False))
            decoder_layers.append(act)
        decoder_layers.append(nn.Dense(self.output_size))
        self.decoder = nn.Sequential(decoder_layers)
    
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = jnp.split(mu_logvar, 2, axis=-1)
        return mu, logvar
    
    def reparameterize(self, rng, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, std.shape)
        return mu + eps * std
    
    def __call__(self, x, rng):
        mu, logvar = self.encode(x)
        z = self.reparameterize(rng, mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def generate(self, rng, n_samples):
        z = random.normal(rng, (n_samples, self.latent_size))
        return self.decoder(z)

class Discriminator(nn.Module):
    input_size: int
    hidden_sizes: Sequence[int] = (256, 128)
    batchnorm: bool = True
    activation: str = 'relu'
    
    def setup(self):
        act = getattr(nn, self.activation)
        
        # Use ModuleList to store layers
        self.layers = nn.ModuleList([
            nn.Dense(self.hidden_sizes[0]),
            act
        ])
        
        # Add intermediate layers with optional batchnorm and dropout
        for i, out_size in enumerate(self.hidden_sizes[1:]):
            self.layers.append(nn.Dense(out_size))
            if self.batchnorm:
                self.layers.append(nn.BatchNorm(use_running_average=False))
            self.layers.append(act)
            if i < len(self.hidden_sizes) - 2:  # Add dropout to all but last layer
                self.layers.append(nn.Dropout(0.2))
        
        # Final layers
        self.final_dense = nn.Dense(1)
        self.final_activation = nn.sigmoid
    
    def __call__(self, x, rng=None, training: bool = True):
        if training and any(isinstance(layer, nn.Dropout) for layer in self.layers) and rng is None:
            raise ValueError("In training mode, rng must be provided for dropout")
        
        dropout_rng = None
        if training and rng is not None:
            dropout_rng = rng
        
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x, deterministic=not training, rng=dropout_rng)
                if dropout_rng is not None:
                    dropout_rng = jax.random.fold_in(dropout_rng, 1)
            else:
                x = layer(x)
        
        x = self.final_dense(x)
        return self.final_activation(x)

def init_weights(rng, module, input_shape):
    if isinstance(module, nn.Dense):
        rng1, rng2 = random.split(rng)
        # Xavier uniform initialization
        w = random.uniform(
            rng1,
            (input_shape[-1], module.features),
            minval=-jnp.sqrt(6.0 / (input_shape[-1] + module.features)),
            maxval=jnp.sqrt(6.0 / (input_shape[-1] + module.features))
        )
        b = random.normal(rng2, (module.features,))
        return {'params': {'kernel': w, 'bias': b}}
    elif isinstance(module, nn.BatchNorm):
        scale = random.normal(rng, (input_shape[-1],))
        return {'params': {'scale': scale, 'bias': jnp.zeros(input_shape[-1])}}
    return {}