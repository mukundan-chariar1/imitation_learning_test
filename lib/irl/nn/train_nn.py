import jax
import jax.numpy as jp
from flax.training import train_state
import optax
from lib.irl.gail import Generator, Discriminator
from lib.irl.loss import D_real_loss_fn, D_fake_loss_fn, G_loss_fn, D_KL
from lib.irl.utils import JAXDataLoader, TrainState
from lib.utils.trajectory import get_observation

import matplotlib.pyplot as plt

def plot_losses(loss_dict):
    plt.figure(figsize=(12, 6))
    for key, losses in loss_dict.items():
        plt.plot(losses, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE-GAN Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@jax.jit
def discriminator_step(state_D, state_G, batch, rng):
    real = batch
    # Split rng for generator, discriminator params, and dropout
    rng, gen_rng, disc_rng, dropout_rng = jax.random.split(rng, 4)
    
    # Generator forward pass
    (fake, mu, logvar), new_model_state = state_G.apply_fn(
        {'params': state_G.params, 'batch_stats': state_G.batch_stats},
        batch, 
        gen_rng,
        mutable=['batch_stats']
    )

    def loss_fn(params_D):
        # Discriminator forward passes with proper RNG handling
        D_real_pred = state_D.apply_fn(
            {'params': params_D}, 
            real, 
            rng=dropout_rng,
            training=True
        )
        D_fake_pred = state_D.apply_fn(
            {'params': params_D}, 
            fake, 
            rng=jax.random.fold_in(dropout_rng, 1),
            training=True
        )
        return D_real_loss_fn(D_real_pred) + D_fake_loss_fn(D_fake_pred)

    grads = jax.grad(loss_fn)(state_D.params)
    new_state_D = state_D.apply_gradients(grads=grads)
    
    # Update generator's batch_stats if they exist
    if new_model_state:
        state_G = state_G.replace(batch_stats=new_model_state['batch_stats'])
    
    # Calculate the actual loss with current parameters for logging (without dropout)
    D_real = state_D.apply_fn({'params': state_D.params}, real, training=False)
    D_fake = state_D.apply_fn({'params': state_D.params}, fake, training=False)
    loss = D_real_loss_fn(D_real) + D_fake_loss_fn(D_fake)
    
    return new_state_D, state_G, loss

@jax.jit
def generator_step(state_G, state_D, batch, rng, beta_kl=1.0):
    # Split rng for generator and discriminator
    rng, gen_rng, disc_rng = jax.random.split(rng, 3)
    
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state_G.batch_stats}
        (recon_x, mu, logvar), new_model_state = state_G.apply_fn(
            variables,
            batch,
            gen_rng,
            mutable=['batch_stats']
        )
        # Discriminator forward pass with dropout
        D_fake = state_D.apply_fn(
            {'params': state_D.params}, 
            recon_x, 
            rng=disc_rng,
            training=True
        )
        recon_loss = G_loss_fn(D_fake)
        kl_loss = D_KL(mu, logvar)
        return recon_loss + beta_kl * kl_loss, (recon_loss, kl_loss, new_model_state)

    (loss, (recon_loss, kl_loss, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state_G.params)
    new_state_G = state_G.apply_gradients(grads=grads)
    if new_model_state:
        new_state_G = new_state_G.replace(batch_stats=new_model_state['batch_stats'])
    return new_state_G, loss, recon_loss, kl_loss

def train_vaegan(generator: Generator,
                 discriminator: Discriminator,
                 train_data: jp.ndarray,
                 latent_size: int,
                 n_epochs: int = 100,
                 batch_size: int = 128,
                 lr: float = 1e-4,
                 rng: jax.random.PRNGKey = jax.random.PRNGKey(0)):

    input_size = train_data.shape[1]
    dataloader = JAXDataLoader(train_data, batch_size=batch_size, rng=rng)

    rng, g_rng, d_rng = jax.random.split(rng, 3)

    # Initialize Generator
    g_rng, init_rng = jax.random.split(g_rng)
    g_variables = generator.init(init_rng, jp.ones((batch_size, input_size)), rng)
    state_G = TrainState.create(
        apply_fn=generator.apply,
        params=g_variables['params'],
        tx=optax.adam(lr),
        batch_stats=g_variables.get('batch_stats', {})  # Add batch_stats to state
    )

    # Initialize Discriminator
    d_rng, init_rng, dropout_rng = jax.random.split(d_rng, 3)
    d_variables = discriminator.init(
        {'params': init_rng, 'dropout': dropout_rng}, 
        jp.ones((batch_size, input_size)),
        training=False  # Disable dropout during initialization
    )
    state_D = TrainState.create(
        apply_fn=discriminator.apply,
        params=d_variables['params'],
        tx=optax.adam(lr),
        batch_stats=d_variables.get('batch_stats', {})
    )

    # Track losses
    loss_history = {
        'D Loss': [],
        'G Loss': [],
        'Recon Loss': [],
        'KL Loss': []
    }

    for epoch in range(n_epochs):
        dataloader = iter(dataloader)
        epoch_d_loss, epoch_g_loss, epoch_recon, epoch_kl = [], [], [], []

        for batch in dataloader:
            rng, step_rng = jax.random.split(rng)
            state_D, state_G, d_loss = discriminator_step(
                state_D, 
                state_G,
                batch, 
                step_rng
            )
            state_G, g_loss, recon_loss, kl_loss = generator_step(
                state_G,
                state_D,
                batch,
                step_rng
            )

            epoch_d_loss.append(d_loss)
            epoch_g_loss.append(g_loss)
            epoch_recon.append(recon_loss)
            epoch_kl.append(kl_loss)

        # Log average epoch losses
        mean_d = jp.mean(jp.stack(epoch_d_loss))
        mean_g = jp.mean(jp.stack(epoch_g_loss))
        mean_recon = jp.mean(jp.stack(epoch_recon))
        mean_kl = jp.mean(jp.stack(epoch_kl))

        print(f"Epoch {epoch+1} | D Loss: {mean_d:.4f} | G Loss: {mean_g:.4f} | Recon: {mean_recon:.4f} | KL: {mean_kl:.4f}")

        loss_history['D Loss'].append(mean_d)
        loss_history['G Loss'].append(mean_g)
        loss_history['Recon Loss'].append(mean_recon)
        loss_history['KL Loss'].append(mean_kl)

    plot_losses(loss_history)
    return state_G, state_D, loss_history

if __name__=='__main__':
    input_size = 200
    latent_size = 16

    # Configs
    generator_config = dict(
        latent_size=latent_size,
        output_size=input_size,
        hidden_sizes=[128, 256, 512],
        batchnorm=True,
        activation='relu',
    )

    discriminator_config = dict(
        input_size=input_size,
        hidden_sizes=[256, 128],
        batchnorm=False,
        activation='relu',
    )

    train_config = dict(
        optimizer_name_G='Adam',
        optimizer_config_G=dict(lr=1e-3, weight_decay=1e-5,),
        lr_scheduler_name_G=None,
        lr_scheduler_config_G=dict(T_max=50000, eta_min=1e-5, last_epoch=-1),

        optimizer_name_D='Adam',
        optimizer_config_D=dict(lr=5e-4, weight_decay=1e-5,),
        lr_scheduler_name_D=None,
        lr_scheduler_config_D=dict(T_max=50000, eta_min=1e-5, last_epoch=-1),

        n_iters=5000,
        batch_size=128,
    )

    # Load and format data
    observations = get_observation()
    train_data = jp.concatenate(observations)
    
    # Determine number of epochs from n_iters and batch size
    dataset_size = train_data.shape[0]
    n_epochs = 1 #train_config['n_iters'] * train_config['batch_size'] // dataset_size

    # Initialize models
    generator = Generator(**generator_config)
    discriminator = Discriminator(**discriminator_config)

    # Train
    rng = jax.random.PRNGKey(0)
    state_G, state_D, loss_history = train_vaegan(
        generator=generator,
        discriminator=discriminator,
        train_data=train_data,
        latent_size=latent_size,
        n_epochs=n_epochs,
        batch_size=train_config['batch_size'],
        lr=train_config['optimizer_config_G']['lr'],  # Assuming same lr for both
        rng=rng
    )