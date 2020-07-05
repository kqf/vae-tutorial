import torch
import pytest

from models.autoencoder import Encoder, Decoder, AutoEncoder


@pytest.fixture
def image_shape():
    return (1, 28, 28)


def test_encoder(image_shape, batch_size=128, latent_size=2):
    enc = Encoder(image_shape, latent_size=latent_size)
    output = enc(torch.zeros(batch_size, *image_shape))
    assert output.shape == (batch_size, latent_size)


def test_decoder(image_shape, batch_size=128, latent_size=2):
    dec = Decoder(image_shape, latent_size=latent_size)
    output = dec(torch.zeros(batch_size, latent_size))
    assert output.shape[0] == batch_size
    assert output.shape[1:] == image_shape


def test_autoencoder(image_shape, batch_size=128, latent_size=2):
    encoder = AutoEncoder(image_shape, latent_size=latent_size)
    batch = torch.zeros(batch_size, *image_shape)
    x_rec, z = encoder(batch)

    assert x_rec.shape == batch.shape
    assert z.shape == (batch_size, latent_size)
