import jax
import jax.numpy as jp
from flax import nnx
import optax

from lib.utils.trajectory import get_observation

from pdb import set_trace as st
from jax.debug import breakpoint as jst

from flax.training import train_state
from typing import Any, Optional

import pickle
import matplotlib.pyplot as plt

class JAXDataLoader:
    def __init__(self, 
                 data: jp.ndarray, 
                 batch_size: int = 128, 
                 rng: jax.random.PRNGKey = jax.random.PRNGKey(0), 
                 shuffle: bool = True,
                 normalize: bool = True):
        
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng

        self.indices = jp.arange(self.data.shape[0])
        self.num_batches = self.data.shape[0] // batch_size
        self.current_batch = 0

        if normalize:
            self.data_mean = self.data.mean(0)
            self.data_std = self.data.std(0)

        if self.shuffle:
            self._shuffle()

    def _shuffle(self):
        self.rng, subkey = jax.random.split(self.rng)
        self.indices = jax.random.permutation(subkey, self.indices)

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        batch = self.data[batch_indices]
        self.current_batch += 1
        return batch
    
    def __len__(self):
        return self.num_batches
    
    def normalize(self, batch, eps=1e-8):
        return (batch-self.data_mean)/(self.data_std+eps)
    
    def unnormalize(self, batch, eps=1e-8):
        return batch*(self.data_std+eps)+self.data_mean
    
class JAXDataLoaderDiff:
    def __init__(self, 
                 data: list, 
                 batch_size: int = 128, 
                 rng: jax.random.PRNGKey = jax.random.PRNGKey(0), 
                 shuffle: bool = True,
                 normalize: bool = True):
        
        self.data = jp.concatenate([jp.concatenate((d[:-1], d[1:]), axis=1) for d in data])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng

        self.indices = jp.arange(self.data.shape[0])
        self.num_batches = self.data.shape[0] // batch_size
        self.current_batch = 0

        if normalize:
            self.data_mean = self.data.mean(0)
            self.data_std = self.data.std(0)

        if self.shuffle:
            self._shuffle()

    def _shuffle(self):
        self.rng, subkey = jax.random.split(self.rng)
        self.indices = jax.random.permutation(subkey, self.indices)

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        batch = self.data[batch_indices]
        self.current_batch += 1
        return batch
    
    def __len__(self):
        return self.num_batches
    
    def normalize(self, batch, eps=1e-8):
        return (batch-self.data_mean)/(self.data_std+eps)
    
    def unnormalize(self, batch, eps=1e-8):
        return batch*(self.data_std+eps)+self.data_mean

def save_config(config: dict, path: str='weights'):
    with open(path, "wb") as f:
        pickle.dump(config, f)

def load_config(path: str='weights'):
    with open(path, "rb") as f:
        config = pickle.load(f)
    return config

def load_model(model_cls, rng, path: str='weights', **kwargs) -> nnx.Module:
    with open(path, "rb") as f:
        state = pickle.load(f)
    model = model_cls(rngs=nnx.Rngs(rng), **kwargs)
    nnx.update(model, state)
    return model

def save_model(model: nnx.Module, path: str='weights'):
    with open(path, "wb") as f:
        state = nnx.state(model)
        pickle.dump(state, f)

class Tracker:
    def __init__(
        self,
        n_iters: int,
        plot_freq: Optional[int] = None,
        figsize: tuple = (18, 6)
    ):
        self.real_scores = []
        self.fake_scores = []
        self.D_losses = []
        self.G_losses = []
        self.recon_losses = []
        self.kl_losses = []

        self.iter = 0
        self.n_iters = n_iters
        self.plot = plot_freq is not None

        if self.plot:
            self.plot_freq = plot_freq
            self.figsize = figsize
            plt.ion()
            self._init_plots()

    def _init_plots(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=self.figsize, sharex=True)

        # Discriminator scores
        self.real_score_curve, = self.ax1.plot([], [], label=r'$D(x)$')
        self.fake_score_curve, = self.ax1.plot([], [], label=r'$D(G(z))$')
        self._setup_plot(self.ax1, 'Discriminator Scores', 'Score', ylim=(0, 1)) #, ylim=(0, 1)

        # Combined loss plot
        self.D_loss_curve, = self.ax2.plot([], [], label='D Loss')
        self.G_loss_curve, = self.ax2.plot([], [], label='G Loss')
        self.recon_loss_curve, = self.ax2.plot([], [], label='Recon Loss')
        self.kl_loss_curve, = self.ax2.plot([], [], label='KL Loss')
        self._setup_plot(self.ax2, 'Losses', 'Loss', ylim=(0, 2))

        self.fig.suptitle('Training Progress', fontsize=14)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.88)
        plt.pause(0.01)

    def _setup_plot(self, ax, title, ylabel, ylim=None):
        ax.set_xlim(0, self.n_iters + 1)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle='--')
        ax.legend(loc='best')

    def update(
        self,
        real_score: float,
        fake_score: float,
        D_loss: float,
        G_loss: float,
        recon_loss: float,
        kl_loss: float
    ):
        self.real_scores.append(real_score)
        self.fake_scores.append(fake_score)
        self.D_losses.append(D_loss)
        self.G_losses.append(G_loss)
        self.recon_losses.append(recon_loss)
        self.kl_losses.append(kl_loss)
        self.iter += 1

        if self.plot and self.iter % self.plot_freq == 0:
            self._update_plots()

    def _update_plots(self):
        x = range(1, self.iter + 1)

        # Update discriminator score curves
        self.real_score_curve.set_data(x, self.real_scores)
        self.fake_score_curve.set_data(x, self.fake_scores)

        # Update loss curves
        self.D_loss_curve.set_data(x, self.D_losses)
        self.G_loss_curve.set_data(x, self.G_losses)
        self.recon_loss_curve.set_data(x, self.recon_losses)
        self.kl_loss_curve.set_data(x, self.kl_losses)

        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self, save: bool = True):
        if self.plot:
            self._update_plots()
            if save:
                self.fig.savefig('fig.png')
            plt.ioff()
            plt.close(self.fig)
        
if __name__=='__main__':
    observations=get_observation()
    dataloader=JAXDataLoader(data=jp.concatenate(observations))