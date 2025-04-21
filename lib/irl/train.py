import jax
import jax.numpy as jp
from flax import nnx
import optax
from typing import Dict, Tuple
from functools import partial

from tqdm import tqdm

from lib.irl.gail import Generator, Discriminator
from lib.irl.loss import D_real_loss_fn, D_fake_loss_fn, G_loss_fn, D_KL
from lib.irl.utils import *
from lib.utils.trajectory import get_observation

from pdb import set_trace as st
from jax.debug import breakpoint as jst

def pretrain_discriminator(discriminator,
                           generator,
                           train_loader,
                           rng_d,
                           rng_g,
                           optimizer_name_D="adamw",
                           optimizer_config_D=dict(learning_rate=1e-3),
                           pretrain_epochs: int=10,
                           ):
    
    iter_pbar = tqdm(range(pretrain_epochs), desc="Training", unit="iter")
    optimizer_D = nnx.Optimizer(discriminator, getattr(optax, optimizer_name_D)(**optimizer_config_D))

    discriminator.train()
    generator.eval()

    @nnx.jit
    def pretrain_step(discriminator: Discriminator, d_optimizer: nnx.Optimizer, x_real, rng_d, rng_g):
        # Generate fake samples
        z = jax.random.normal(rng_g, (x_real.shape[0], generator.latent_size))  # make sure latent_dim is defined
        x_fake = generator.decode(z)

        def d_loss_fn(discriminator: Discriminator):
            d_real = discriminator(x_real)
            d_fake = discriminator(x_fake)

            d_real_loss = D_real_loss_fn(d_real)
            d_fake_loss = D_fake_loss_fn(d_fake)

            return (d_real_loss+d_fake_loss)/2

        d_loss, grads = nnx.value_and_grad(d_loss_fn)(discriminator)
        d_optimizer.update(grads)

        return d_loss

    # Training loop
    for epoch in range(pretrain_epochs):
        for x_real in train_loader:
            rng_d, step_rng_d = jax.random.split(rng_d)
            rng_g, step_rng_g = jax.random.split(rng_g)
            loss = pretrain_step(discriminator, optimizer_D, x_real, step_rng_d, step_rng_g)

            iter_pbar.set_postfix({
                'Loss': float(loss)
            })

        iter_pbar.update(1)



def train_GAN(generator,
              discriminator,
              train_dataset,
              rng,
              optimizer_name_G="adamw",
              optimizer_config_G=dict(learning_rate=1e-3),
              optimizer_name_D="adamw",
              optimizer_config_D=dict(learning_rate=1e-3),
              optimizer_name_D_pretrain="adamw",
              optimizer_config_D_pretrain=dict(learning_rate=1e-3),
              pretrain_epochs=10,
              epochs=10,
              batch_size=64,
              recon_weight=1.0,
              kl_weight=0.01,
              adv_weight=0.001,
              ):
    train_loader = JAXDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    tracker = Tracker(n_iters=epochs*len(train_loader), plot_freq=1000)

    # rng, rng_d, rng_g = jax.random.split(rng, 3)
    # pretrain_discriminator(discriminator, generator, train_loader, rng_d, rng_g, optimizer_name_D_pretrain, optimizer_config_D_pretrain, pretrain_epochs)
    
    iter_pbar = tqdm(range(epochs), desc="Training", unit="iter")

    optimizer_G = nnx.Optimizer(generator, getattr(optax, optimizer_name_G)(**optimizer_config_G))
    optimizer_D = nnx.Optimizer(discriminator, getattr(optax, optimizer_name_D)(**optimizer_config_D))

    @nnx.jit
    def discriminator_step(generator: Generator, d_optimizer: nnx.Optimizer, x_real, rng):
        generator.eval()
        discriminator=d_optimizer.model
        discriminator.train()
        
        def d_loss_fn(discriminator: Discriminator):
            d_real = discriminator(x_real)
            d_real_loss = D_real_loss_fn(d_real)

            x_recon, mu, logvar = generator(x_real, rng=rng)
            z = jax.random.normal(rng, shape=(x_real.shape[0], generator.latent_size))
            x_fake = generator.decode(z)

            d_fake_recon = discriminator(x_recon)
            d_fake_gen = discriminator(x_fake)
            d_fake_loss = (D_fake_loss_fn(d_fake_recon) + D_fake_loss_fn(d_fake_gen)) / 2.0

            d_loss = (d_real_loss + d_fake_loss) / 2.0
            return d_loss

        d_loss, grads = nnx.value_and_grad(d_loss_fn)(discriminator)
        d_optimizer.update(grads)

        discriminator.eval()
        d_real = discriminator(x_real)
        d_real_loss = D_real_loss_fn(d_real)

        x_recon, mu, logvar = generator(x_real, rng=rng)
        z = jax.random.normal(rng, shape=(x_real.shape[0], generator.latent_size))
        x_fake = generator.decode(z)

        d_fake_recon = discriminator(x_recon)
        d_fake_gen = discriminator(x_fake)
        d_fake_loss = (D_fake_loss_fn(d_fake_recon) + D_fake_loss_fn(d_fake_gen)) / 2.0

        return d_loss, d_real_loss, d_fake_loss

    @nnx.jit
    def generator_step(discriminator: Discriminator, g_optimizer: nnx.Optimizer, x_real, rng):
        discriminator.eval()
        generator=g_optimizer.model
        generator.train()

        def g_loss_fn(generator: Generator):
            x_recon, mu, logvar = generator(x_real, rng=rng)
            recon_loss = jp.mean((x_recon - x_real) ** 2)
            kl_loss = D_KL(mu, logvar)
            
            d_recon = discriminator(x_recon)
            adv_loss = G_loss_fn(d_recon)

            total_loss = (
                recon_weight * recon_loss
                + kl_weight * kl_loss
                + adv_weight * adv_loss
            )
            return total_loss

        g_loss, grads = nnx.value_and_grad(g_loss_fn)(generator)
        g_optimizer.update(grads)

        generator.eval()
        x_recon, mu, logvar = generator(x_real, rng=rng)
        recon_loss = jp.mean((x_recon - x_real) ** 2)
        kl_loss = D_KL(mu, logvar)
        
        d_recon = discriminator(x_recon)
        adv_loss = G_loss_fn(d_recon)
        return g_loss, recon_loss, kl_loss, adv_loss

    for epoch in range(epochs):
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        epoch_D_real = 0.0
        epoch_D_fake = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        batch_count = 0

        for i, x_real in enumerate(train_loader):
            rng, rng_d, rng_g = jax.random.split(rng, 3)

            # Training steps
            d_loss, d_real_loss, d_fake_loss = discriminator_step(generator, optimizer_D, x_real, rng_d)
            g_loss, recon_loss, kl_loss, adv_loss = generator_step(discriminator, optimizer_G, x_real, rng_g)

            # Accumulate losses
            epoch_D_loss += float(d_loss)
            epoch_G_loss += float(g_loss)
            epoch_D_real += float(d_real_loss)
            epoch_D_fake += float(d_fake_loss)
            epoch_recon += float(recon_loss)
            epoch_kl += float(kl_loss)
            batch_count += 1

            # Update progress bar
            iter_pbar.set_postfix({
                'D_loss': float(epoch_D_loss/batch_count),
                'G_loss': float(epoch_G_loss/batch_count),
                'D_real': float(epoch_D_real/batch_count),
                'D_fake': float(epoch_D_fake/batch_count),
                'Recon_loss': float(epoch_recon/batch_count),
                'KL_loss': float(epoch_kl/batch_count)
            })

            tracker.update(
                real_score=d_real_loss,
                fake_score=d_fake_loss,
                D_loss=d_loss,
                G_loss=g_loss,
                recon_loss=recon_loss,
                kl_loss=kl_loss
            )

        iter_pbar.update(1)

    tracker.close(True)

    save_model(generator, "weights/generator.pkl")
    save_model(discriminator, "weights/discriminator.pkl")


if __name__ == '__main__':
    rng = jax.random.PRNGKey(42)
    input_size = 47
    latent_size = 3

    generator_config = dict(
        latent_size=latent_size,
        output_size=input_size,
        hidden_sizes=[64, 64],
        batchnorm=False,
        activation='relu',
    )

    discriminator_config = dict(
        input_size=input_size,
        hidden_sizes=[128, 128],
        batchnorm=True,
        activation='relu',
    )

    save_config(generator_config, 'weights/generator_config.pkl')
    save_config(discriminator_config, 'weights/discriminator_config.pkl')

    train_config = dict(
        optimizer_name_G='AdamW',
        optimizer_config_G=dict(learning_rate=1e-3, weight_decay=1e-5,),
        lr_scheduler_name_G=None,
        lr_scheduler_config_G=dict(T_max=50000, eta_min=1e-5, last_epoch=-1),

        optimizer_name_D='AdamW',
        optimizer_config_D=dict(learning_rate=5e-5, weight_decay=1e-5,),
        optimizer_name_D_pretrain='AdamW',
        optimizer_config_D_pretrain=dict(learning_rate=1e-7, weight_decay=1e-5,),
        lr_scheduler_name_D=None,
        lr_scheduler_config_D=dict(T_max=50000, eta_min=1e-5, last_epoch=-1),

        pretrain_epochs=1,
        epochs=50,
        batch_size=256,
    )

    rng, gen_rng, disc_rng = jax.random.split(rng, 3)
    generator = Generator(rngs=nnx.Rngs(gen_rng), **generator_config)
    discriminator = Discriminator(rngs=nnx.Rngs(disc_rng), **discriminator_config)

    observations = get_observation()
    train_data = jp.concatenate(observations)

    train_GAN(
        generator=generator,
        discriminator=discriminator,
        train_dataset=train_data,
        rng=rng,
        optimizer_name_G=train_config["optimizer_name_G"].lower(),
        optimizer_config_G=train_config["optimizer_config_G"],
        optimizer_name_D=train_config["optimizer_name_D"].lower(),
        optimizer_config_D=train_config["optimizer_config_D"],
        epochs=train_config["epochs"],
        batch_size=train_config["batch_size"],
        recon_weight=1,
        kl_weight=0.01,
        adv_weight=0.001,
        pretrain_epochs=1,
    )

    # generator = load_model(Generator, gen_rng, "weights/generator.pkl", **generator_config)
    # discriminator = load_model(Discriminator, disc_rng, "weights/discriminator.pkl", **discriminator_config)



    