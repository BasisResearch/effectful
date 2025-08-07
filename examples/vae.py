import argparse
import os

import flax.nnx as nnx
import jax
import optax
import torch  # to download the MNIST dataset
from effectful.handlers import numpyro as dist
from effectful.handlers.jax import bind_dims, unbind_dims
from effectful.handlers.jax import numpy as jnp
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from jax import random
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST

from weighted.handlers.jax import DenseTensorFold, log_prob, sample
from weighted.handlers.optimization import simplify_normals_intp
from weighted.ops.distribution import kl_divergence
from weighted.ops.sugar import Sum

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_PATH, "data")


latent_dim = defop(jax.Array, name="latent_dim")
batch_dim = defop(jax.Array, name="batch_dim")
img_x_dim = defop(jax.Array, name="img_x_dim")
img_y_dim = defop(jax.Array, name="img_y_dim")


class VAE(nnx.Module):
    def __init__(self, hidden_size: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(28 * 28, hidden_size, rngs=rngs)
        self.fc2_loc = nnx.Linear(hidden_size, args.nb_latents, rngs=rngs)
        self.fc2_scale = nnx.Linear(hidden_size, args.nb_latents, rngs=rngs)
        self.fc3 = nnx.Linear(args.nb_latents, hidden_size, rngs=rngs)
        self.fc4 = nnx.Linear(hidden_size, 28 * 28, rngs=rngs)

    def prior(self):
        """p(Z)  prior distribution over latents"""
        prior_mean = unbind_dims(jnp.zeros(args.nb_latents), latent_dim)
        prior_std = unbind_dims(jnp.ones(args.nb_latents), latent_dim)
        return dist.Normal(prior_mean, prior_std)

    def encode(self, image):
        """q(Z | x=image)  posterior distribution over latents conditioned on image"""
        image = image.reshape(image.shape[:-2] + (-1,))
        h1 = jax.nn.relu(self.fc1(image))
        loc = self.fc2_loc(h1)
        loc = unbind_dims(loc, batch_dim, latent_dim)
        scale = jnp.exp(self.fc2_scale(h1))
        scale = unbind_dims(scale, batch_dim, latent_dim)
        return dist.Normal(loc, scale)

    def decode(self, z):
        """p(X | Z=z)  image distribution conditioned on latents"""
        z = bind_dims(z, batch_dim, latent_dim)
        h3 = jax.nn.relu(self.fc3(z))
        logits = self.fc4(h3)
        logits = logits.reshape(logits.shape[:-1] + (28, 28))
        # logits = unbind_dims(logits, batch_dim, img_x_dim, img_y_dim)
        return dist.BernoulliLogits(logits=logits)

    def __call__(self, image, key):
        q = self.encode(image)
        z = sample(key, q, ())
        p = self.decode(z)
        return p, q


def elbo_loss(model, x, key, beta=2.0):
    """Negative ELBO loss: reconstruction_loss + β * KL_loss"""
    p, q = model(x, key)

    image_streams = {img_x_dim: jnp.arange(28), img_y_dim: jnp.arange(28)}
    latent_stream = {latent_dim: jnp.arange(args.nb_latents)}
    batch_stream = {batch_dim: jnp.arange(x.shape[0])}

    log_prob_x = unbind_dims(log_prob(p, x), batch_dim, img_x_dim, img_y_dim)
    reconstruction_loss = -Sum(image_streams, log_prob_x)
    kl_loss = Sum(latent_stream, kl_divergence(q, model.prior()))
    elbo = reconstruction_loss + beta * kl_loss
    return Sum(batch_stream, elbo) / x.shape[0]


elbo_loss_grad = nnx.jit(nnx.value_and_grad(elbo_loss))


def train(dataset_loader, model, optimizer, key):
    total_loss = 0
    for data, _ in dataset_loader:
        data = jnp.array(data[:, 0, :, :])
        key, subkey = random.split(key)

        with handler(DenseTensorFold()), handler(simplify_normals_intp):
            loss, grads = elbo_loss_grad(model, data, subkey)
        total_loss += loss
        optimizer.update(grads)
    return total_loss, key


def main(args):
    key = random.PRNGKey(42)
    key, model_key, eval_key, sample_key = random.split(key, 4)

    model = VAE(hidden_size=512, rngs=nnx.Rngs(model_key))
    train_loader = torch.utils.data.DataLoader(
        MNIST(DATA_PATH, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    optimizer = optax.adamw(learning_rate=1e-3)
    optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    for epoch in range(args.nb_epochs):
        print(f"### epoch {epoch} ###")
        loss, key = train(train_loader, model, optimizer, key)
        print(f"loss={loss:.4f}")

    model.eval()
    test_dataset = MNIST(
        DATA_PATH, train=False, download=True, transform=transforms.ToTensor()
    )
    plot_reconstructions(model, test_dataset, eval_key, n_images=5)
    plot_samples(model, sample_key, n_samples=16)


def plot_samples(model, key, n_samples=16):
    """Plot samples generated from the VAE"""
    samples = sample(key, model.prior(), (n_samples,))
    samples = unbind_dims(samples, batch_dim)
    samples = model.decode(samples).probs
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Generated Samples", fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_reconstructions(model, test_dataset, key, n_images=5):
    """Plot original vs reconstructed images"""
    fig, axes = plt.subplots(2, n_images, figsize=(12, 4))
    fig.suptitle("Original (top) vs Reconstructed (bottom)", fontsize=14)

    for i in range(n_images):
        img, label = test_dataset[i]
        reconstructed = model(img, key)[0].probs
        # Original
        axes[0, i].imshow(img.squeeze(), cmap="gray")
        axes[0, i].set_title(f"Label: {label}")
        axes[0, i].axis("off")
        # Reconstructed
        axes[1, i].imshow(reconstructed.squeeze(), cmap="gray")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE MNIST Example")
    parser.add_argument("-n", "--nb-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nb-latents", type=int, default=32)
    args = parser.parse_args()
    main(args)
