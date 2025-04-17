import jax.numpy as jnp
from jax import nn as jnn

def D_real_loss_fn(
        D_real: jnp.ndarray,  # (batch_size, 1)
) -> jnp.ndarray:  # ()
    """
    D_real is D(x), the discriminator's output when fed with real images
    We want this to be close to 1, because the discriminator should recognize real images
    """
    target = jnp.ones_like(D_real)  # *0.99 for label smoothing if desired
    return jnn.binary_cross_entropy(D_real, target)

def D_fake_loss_fn(
        D_fake: jnp.ndarray,  # (batch_size, 1)
) -> jnp.ndarray:  # ()
    """
    D_fake is D(G(z)), the discriminator's output when fed with generated images
    We want this to be close to 0, because the discriminator should not be fooled
    """
    target = jnp.zeros_like(D_fake)  # or 0.01 for label smoothing
    return jnn.binary_cross_entropy(D_fake, target)

def G_loss_fn(
        D_fake: jnp.ndarray,  # (batch_size, 1)
) -> jnp.ndarray:  # ()
    """
    D_fake is D(G(z)), the discriminator's output when fed with generated images
    We want this to be close to 1, because the generator wants to fool the discriminator
    """
    target = jnp.ones_like(D_fake)  # *0.99 for label smoothing
    return jnn.binary_cross_entropy(D_fake, target)

def D_KL(
        mu: jnp.ndarray,  # shape (batch_size, latent_size)
        logvar: jnp.ndarray,  # shape (batch_size, latent_size)
) -> jnp.ndarray:  # shape ()
    """
    Compute the KL divergence between N(mu, var) and N(0, 1)
    Then average over batch and latent dimensions
    
    mu: mean of q(z|x)
    logvar: Logarithm of variance of q(z|x)
    """
    var = jnp.exp(logvar)
    kl = 0.5 * jnp.mean(var + mu**2 - 1 - logvar)  # Mean over all dimensions
    # Alternative implementation that first sums over latent dims:
    # kl = 0.5 * jnp.mean(jnp.sum(var + mu**2 - 1 - logvar, axis=-1))
    return kl