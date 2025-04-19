import jax
import jax.numpy as jp
from typing import Sequence, Optional
from flax import nnx

class Generator(nnx.Module):
    def __init__(self, 
                 latent_size: int,
                 output_size: int,
                 hidden_sizes: Sequence[int] = (128, 256, 256),
                 batchnorm: bool = True,
                 activation: str = 'relu',
                 rngs: nnx.Rngs = None):
        super().__init__()
        self.latent_size = latent_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.batchnorm = batchnorm
        self.activation = getattr(nnx, activation)
        
        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Encoder layers
        self.encoder_layers = [
            nnx.Linear(output_size, self.hidden_sizes[-1], rngs=rngs),
            self.activation
        ]
        
        for in_size, out_size in reversed(list(zip(self.hidden_sizes[1:], self.hidden_sizes[:-1]))):
            self.encoder_layers.extend([
                nnx.Linear(in_size, out_size, rngs=rngs),
                nnx.BatchNorm(out_size, use_running_average=False, rngs=rngs) if self.batchnorm else None,
                self.activation
            ])
        
        self.encoder_layers.append(nnx.Linear(self.hidden_sizes[0], self.latent_size * 2, rngs=rngs))
        
        # Decoder layers
        self.decoder_layers = [
            nnx.Linear(latent_size, self.hidden_sizes[0], rngs=rngs),
            self.activation
        ]
        
        for in_size, out_size in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:]):
            self.decoder_layers.extend([
                nnx.Linear(in_size, out_size, rngs=rngs),
                nnx.BatchNorm(out_size, use_running_average=False, rngs=rngs) if self.batchnorm else None,
                self.activation
            ])
        
        self.decoder_layers.append(nnx.Linear(self.hidden_sizes[-1], self.output_size, rngs=rngs))
    
    # Rest of the methods remain the same...
    def encode(self, x):
        for layer in self.encoder_layers:
            if layer is not None:
                x = layer(x)
        return x
    
    def decode(self, z):
        x = z
        for layer in self.decoder_layers:
            if layer is not None:
                x = layer(x)
        return x
    
    def __call__(self, x, rng):
        mu_logvar = self.encode(x)
        mu, logvar = jp.split(mu_logvar, 2, axis=-1)
        z = self.reparameterize(rng, mu, logvar)
        return self.decode(z), mu, logvar
    
    def reparameterize(self, rng, mu, logvar):
        std = jp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, std.shape)
        return mu + eps * std
    
    def generate(self, rng, n_samples):
        z = jax.random.normal(rng, (n_samples, self.latent_size))
        return self.decode(z)

class Discriminator(nnx.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: Sequence[int] = (256, 128),
                 batchnorm: bool = True,
                 activation: str = 'relu',
                 rngs: nnx.Rngs = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batchnorm = batchnorm
        self.activation = getattr(nnx, activation)
        
        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Initialize layers
        self.layers = [
            nnx.Linear(input_size, self.hidden_sizes[0], rngs=rngs),
            self.activation
        ]
        
        # Add intermediate layers
        for in_size, out_size in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:]):
            self.layers.extend([
                nnx.Linear(in_size, out_size, rngs=rngs),
                nnx.BatchNorm(out_size, use_running_average=False, rngs=rngs) if self.batchnorm else None,
                self.activation,
                nnx.Dropout(0.2, rngs=rngs)
            ])
        
        # Final layers
        self.final_dense = nnx.Linear(self.hidden_sizes[-1], 1, rngs=rngs)
    
    def __call__(self, x):
        for layer in self.layers:
            if layer is not None: 
                x = layer(x)
        return self.final_dense(x)

def init_weights(rng, module, input_shape):
    if isinstance(module, nnx.Linear):
        rng1, rng2 = jax.random.split(rng)
        # Xavier uniform initialization
        w = jax.random.uniform(
            rng1,
            (input_shape[-1], module.features),
            minval=-jp.sqrt(6.0 / (input_shape[-1] + module.features)),
            maxval=jp.sqrt(6.0 / (input_shape[-1] + module.features))
        )
        b = jax.random.normal(rng2, (module.features,))
        return {'params': {'kernel': w, 'bias': b}}
    elif isinstance(module, nnx.BatchNorm):
        scale = jax.random.normal(rng, (input_shape[-1],))
        return {'params': {'scale': scale, 'bias': jp.zeros(input_shape[-1])}}
    return {}