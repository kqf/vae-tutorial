import torch
import pytest

from models.vae import VAE
from models.vae import kl_gaussian, ELBO


@pytest.fixture
def image_shape():
    return (1, 28, 28)


def test_vae(image_shape, batch_size=128, hid_dim=2):
    x = torch.zeros(batch_size, *image_shape)

    encoder = VAE(image_shape, hid_dim)
    mean_image, sampled_image, logits, z, mean, stddev = encoder(x)

    assert mean_image.shape == x.shape
    assert sampled_image.shape == x.shape
    assert logits.shape == x.shape

    assert z.shape == (batch_size, hid_dim)
    assert mean.shape == (batch_size, hid_dim)
    assert stddev.shape == (batch_size, hid_dim)


def test_kl():
    mean_vectors = torch.tensor([
        [0.25, 0.0],  # Example 1.
        [0.5, 0.0],  # Example 2.
    ])

    variance = torch.tensor([
        [1, 1],  # Example 1.
        [0.1, 1],  # Example 2.
    ])
    expected = torch.tensor([
        0.03125,  # Example 1.
        0.8262926,  # Example 2.
    ])
    output = kl_gaussian(mean_vectors, variance)

    assert torch.allclose(output, expected)


def test_elbo(image_shape):
    criterion = ELBO(reduction="sum")
    y_pred = (
        None,
        None,
        torch.ones(*image_shape),
        None,
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, 1.0]])
    )
    y_true = torch.ones(*image_shape)
    output = criterion(y_pred, y_true)
    assert torch.allclose(output, torch.tensor(245.59708))
