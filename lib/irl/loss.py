import jax.numpy as jp
import optax

def D_real_loss_fn(logits_real: jp.ndarray) -> jp.ndarray:
    # labels = jp.ones_like(logits_real)
    labels = jp.full_like(logits_real, 0.9)
    loss = optax.sigmoid_binary_cross_entropy(logits_real, labels)
    return jp.mean(loss)

def D_fake_loss_fn(logits_fake: jp.ndarray) -> jp.ndarray:
    labels = jp.zeros_like(logits_fake)
    loss = optax.sigmoid_binary_cross_entropy(logits_fake, labels)
    return jp.mean(loss)

def G_loss_fn(logits_fake: jp.ndarray) -> jp.ndarray:
    # labels = jp.ones_like(logits_fake)
    labels = jp.full_like(logits_fake, 0.9)
    loss = optax.sigmoid_binary_cross_entropy(logits_fake, labels)
    return jp.mean(loss)

def D_KL(mu: jp.ndarray, logvar: jp.ndarray) -> jp.ndarray:
    return 0.5 * jp.mean(jp.exp(logvar) + mu**2 - 1. - logvar)
