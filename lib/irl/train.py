import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
from functools import partial
import numpy as np

def train_GAN(
    generator,
    discriminator,
    train_dataset,
    rng,
    plot_freq=100,
    optimizer_name_G="adam",
    optimizer_config_G={"learning_rate": 1e-3},
    lr_scheduler_name_G=None,
    lr_scheduler_config_G={},
    optimizer_name_D="adam",
    optimizer_config_D={"learning_rate": 1e-3},
    lr_scheduler_name_D=None,
    lr_scheduler_config_D={},
    n_iters=10000,
    batch_size=64,
    recon_weight=1.0,
    kl_weight=0.01,
    adv_weight=0.001,
):
    # Initialize models
    rng, init_rng_G, init_rng_D = jax.random.split(rng, 3)
    
    # Create sample input for initialization
    sample_input = jnp.ones((batch_size, *train_dataset[0][0].shape))
    
    # Initialize generator and discriminator
    generator_params = generator.init(init_rng_G, sample_input, rng)
    discriminator_params = discriminator.init(init_rng_D, sample_input)

    # Create optimizers
    if lr_scheduler_name_G:
        schedule = getattr(optax, lr_scheduler_name_G)(**lr_scheduler_config_G)
        optimizer_G = optax.chain(
            getattr(optax, optimizer_name_G)(**optimizer_config_G),
            schedule
        )
    else:
        optimizer_G = getattr(optax, optimizer_name_G)(**optimizer_config_G)

    if lr_scheduler_name_D:
        schedule = getattr(optax, lr_scheduler_name_D)(**lr_scheduler_config_D)
        optimizer_D = optax.chain(
            getattr(optax, optimizer_name_D)(**optimizer_config_D),
            schedule
        )
    else:
        optimizer_D = getattr(optax, optimizer_name_D)(**optimizer_config_D)

    # Create training states
    state_G = train_state.TrainState.create(
        apply_fn=generator.apply,
        params=generator_params,
        tx=optimizer_G
    )
    
    state_D = train_state.TrainState.create(
        apply_fn=discriminator.apply,
        params=discriminator_params,
        tx=optimizer_D
    )

    # Create data loader (simplified version)
    def data_loader(dataset, batch_size, rng):
        while True:
            rng, shuffle_rng = jax.random.split(rng)
            idx = jax.random.permutation(shuffle_rng, len(dataset))
            for i in range(0, len(dataset), batch_size):
                batch = [dataset[int(j)] for j in idx[i:i+batch_size]]
                x_real = jnp.stack([x for x, _ in batch])
                yield x_real

    loader = data_loader(train_dataset, batch_size, rng)

    # Define loss functions
    def compute_D_loss(params_D, params_G, x_real, rng):
        rng, fake_rng = jax.random.split(rng)
        
        # Forward pass through generator
        x_recon, mu, logvar = generator.apply(params_G, x_real, rng)
        z = jax.random.normal(fake_rng, (x_real.shape[0], generator.latent_size))
        x_fake = generator.apply(params_G, method='decode')(params_G, z)
        
        # Discriminator outputs
        d_real = discriminator.apply(params_D, x_real)
        d_fake_recon = discriminator.apply(params_D, x_recon)
        d_fake_gen = discriminator.apply(params_D, x_fake)
        
        # Loss calculations
        d_real_loss = D_real_loss_fn(d_real)
        d_fake_loss = (D_fake_loss_fn(d_fake_recon) + D_fake_loss_fn(d_fake_gen)) / 2
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        return d_loss, (d_real.mean(), d_fake_recon.mean())

    def compute_G_loss(params_G, params_D, x_real, rng):
        x_recon, mu, logvar = generator.apply(params_G, x_real, rng)
        
        # Loss components
        recon_loss = jnp.mean((x_recon - x_real) ** 2)
        kl_loss = D_KL(mu, logvar)
        d_recon = discriminator.apply(params_D, x_recon)
        adv_loss = G_loss_fn(d_recon)
        
        total_loss = recon_weight * recon_loss + kl_weight * kl_loss + adv_weight * adv_loss
        return total_loss, (recon_loss, kl_loss, adv_loss)

    # Training step functions
    @partial(jax.jit, static_argnums=(3,))
    def train_step_D(state_G, state_D, x_real, rng):
        grad_fn = jax.value_and_grad(compute_D_loss, has_aux=True)
        (d_loss, (d_real_score, d_fake_score)), grads = grad_fn(
            state_D.params, state_G.params, x_real, rng
        )
        state_D = state_D.apply_gradients(grads=grads)
        return state_D, d_loss, d_real_score, d_fake_score

    @partial(jax.jit, static_argnums=(3,))
    def train_step_G(state_G, state_D, x_real, rng):
        grad_fn = jax.value_and_grad(compute_G_loss, has_aux=True)
        (g_loss, (recon_loss, kl_loss, adv_loss)), grads = grad_fn(
            state_G.params, state_D.params, x_real, rng
        )
        state_G = state_G.apply_gradients(grads=grads)
        return state_G, g_loss, recon_loss, kl_loss, adv_loss

    # Training loop
    tracker = VAEGAN_Tracker(n_iters, plot_freq)
    iter_pbar = tqdm(range(n_iters), desc="Training", unit="iter")

    for iter in iter_pbar:
        x_real = next(loader)
        rng, step_rng_D, step_rng_G = jax.random.split(rng, 3)

        # Train Discriminator
        state_D, d_loss, d_real_score, d_fake_score = train_step_D(
            state_G, state_D, x_real, step_rng_D
        )

        # Train Generator
        state_G, g_loss, recon_loss, kl_loss, adv_loss = train_step_G(
            state_G, state_D, x_real, step_rng_G
        )

        # Update tracker
        if iter % plot_freq == 0:
            rng, sample_rng = jax.random.split(rng)
            z_sample = jax.random.normal(sample_rng, (24, generator.latent_size))
            gen_samples = generator.apply(state_G.params, method='decode')(state_G.params, z_sample)
            
            idx = np.random.randint(0, x_real.shape[0])
            sample_real = x_real[idx]
            sample_recon = generator.apply(state_G.params, x_real[None, idx], rng)[0][0]
            
            tracker.update_vaegan(
                real_score=d_real_score,
                fake_score=d_fake_score,
                D_loss=d_loss,
                G_loss=g_loss,
                recon_loss=recon_loss,
                kl_loss=kl_loss,
                x_real=sample_real[None],
                x_recon=sample_recon[None],
            )
            tracker.get_samples(gen_samples)

    return state_G, state_D