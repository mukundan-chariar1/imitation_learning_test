import jax
import jax.numpy as jp
from typing import Sequence, Optional
from flax import nnx

from flax.nnx.nn.recurrent import *

# from lib.irl.recurrent import *

from pdb import set_trace as st
from jax.debug import breakpoint as jst

class StepGenerator(nnx.Module):
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
            nnx.BatchNorm(self.hidden_sizes[-1], use_running_average=False, rngs=rngs) if self.batchnorm else None,
            self.activation
        ]
        
        for i, (in_size, out_size) in enumerate(reversed(list(zip(self.hidden_sizes[1:], self.hidden_sizes[:-1])))):
            self.encoder_layers.extend([
                nnx.Linear(in_size, out_size, rngs=rngs),
                nnx.BatchNorm(out_size, use_running_average=False, rngs=rngs) if (self.batchnorm and i==len(self.hidden_sizes)-2) else None,
                self.activation
            ])
        
        self.encoder_layers.append(nnx.Linear(self.hidden_sizes[0], self.latent_size * 2, rngs=rngs))
        
        # Decoder layers
        self.decoder_layers = [
            nnx.Linear(latent_size, self.hidden_sizes[0], rngs=rngs),
            nnx.BatchNorm(self.hidden_sizes[0], use_running_average=False, rngs=rngs) if self.batchnorm else None,
            self.activation
        ]
        
        for i, (in_size, out_size) in enumerate(zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])):
            self.decoder_layers.extend([
                nnx.Linear(in_size, out_size, rngs=rngs),
                nnx.BatchNorm(out_size, use_running_average=False, rngs=rngs) if (self.batchnorm and i==len(self.hidden_sizes)-2) else None,
                self.activation
            ])
        
        self.decoder_layers.append(nnx.Linear(self.hidden_sizes[-1], self.output_size, rngs=rngs))

        self.step_decoder_layers = [
            nnx.Linear(latent_size, self.hidden_sizes[0], rngs=rngs),
            nnx.BatchNorm(self.hidden_sizes[0], use_running_average=False, rngs=rngs) if self.batchnorm else None,
            self.activation
        ]
        
        for i, (in_size, out_size) in enumerate(zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])):
            self.step_decoder_layers.extend([
                nnx.Linear(in_size, out_size, rngs=rngs),
                nnx.BatchNorm(out_size, use_running_average=False, rngs=rngs) if (self.batchnorm and i==len(self.hidden_sizes)-2) else None,
                self.activation
            ])
        
        self.step_decoder_layers.append(nnx.Linear(self.hidden_sizes[-1], self.output_size, rngs=rngs))
    
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
    
    def step_decode(self, z):
        x = z
        for layer in self.step_decoder_layers:
            if layer is not None:
                x = layer(x)
        return x
    
    def __call__(self, x, rng):
        mu_logvar = self.encode(x)
        mu, logvar = jp.split(mu_logvar, 2, axis=-1)
        z = self.reparameterize(rng, mu, logvar)
        return self.decode(z), self.step_decode(z), mu, logvar
    
    def reparameterize(self, rng, mu, logvar):
        std = jp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, std.shape)
        return mu + eps * std
    
    def generate(self, rng, n_samples):
        z = jax.random.normal(rng, (n_samples, self.latent_size))
        return self.decode(z), self.step_decode(z)
    
class StepDiscriminator(nnx.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: Sequence[int] = (256, 128),
                 rnn_hidden_size: int = 64,
                 batchnorm: bool = True,
                 activation: str = 'relu',
                 use_lstm: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batchnorm = batchnorm
        self.activation = getattr(nnx, activation)
        self.rnn_hidden_size = rnn_hidden_size
        
        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(0)

        forward_rnn = RNN(
            GRUCell(in_features=input_size, hidden_features=rnn_hidden_size, rngs=rngs),
            time_major=False,
            return_carry=False
        )
        backward_rnn = RNN(
            GRUCell(in_features=input_size, hidden_features=rnn_hidden_size, rngs=rngs),
            time_major=False,
            return_carry=False,
            reverse=True,
            keep_order=True
        )

        self.birnn = Bidirectional(forward_rnn=forward_rnn, backward_rnn=backward_rnn)

        # Output of bidirectional GRU is 2 * hidden_size
        rnn_output_size = 2 * rnn_hidden_size
        
        # Initialize layers
        self.layers = [
            nnx.Linear(rnn_output_size, self.hidden_sizes[0], rngs=rngs),
            nnx.BatchNorm(self.hidden_sizes[0], use_running_average=False, rngs=rngs) if self.batchnorm else None,
            self.activation
        ]
        
        # Add intermediate layers
        for i, (in_size, out_size) in enumerate(zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])):
            self.layers.extend([
                nnx.Linear(in_size, out_size, rngs=rngs),
                nnx.BatchNorm(out_size, use_running_average=False, rngs=rngs) if (self.batchnorm and i==len(self.hidden_sizes)-2) else None,
                self.activation,
                nnx.Dropout(0.2, rngs=rngs)
            ])
        
        # Final layers
        self.final_dense = nnx.Linear(self.hidden_sizes[-1], 1, rngs=rngs)
    
    def __call__(self, x):
        x = self.birnn(x)
        x = x.mean(axis=1)

        for layer in self.layers:
            if layer is not None: 
                x = layer(x)
        return self.final_dense(x)