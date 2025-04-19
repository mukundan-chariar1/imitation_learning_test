import jax
import jax.numpy as jp
import os

from lib.utils.trajectory import get_observation

from pdb import set_trace as st
from jax.debug import breakpoint as jst

from flax.training import train_state
from typing import Any

class TrainState(train_state.TrainState):
    batch_stats: Any = None  # For batch normalization statistics

class JAXDataLoader:
    def __init__(self, 
                 data: jp.ndarray, 
                 batch_size: int = 128, 
                 rng: jax.random.PRNGKey = jax.random.PRNGKey(0), 
                 shuffle: bool = True):
        
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng

        self.indices = jp.arange(self.data.shape[0])
        self.num_batches = self.data.shape[0] // batch_size
        self.current_batch = 0

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

if __name__=='__main__':
    observations=get_observation()
    dataloader=JAXDataLoader(data=jp.concatenate(observations))