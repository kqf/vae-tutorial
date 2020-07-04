import skorch
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from operator import mul
from functools import reduce


class Encoder(torch.nn.Module):
    def __init__(self, input_shape, hid_size=512, latent_size=10):
        super().__init__()
        flat_features = reduce(mul, input_shape, 1)
        self._hid = torch.nn.Sequential(
            torch.nn.Linear(flat_features, hid_size),
            torch.nn.ReLU(),
        )
        self._mean = torch.nn.Linear(hid_size, latent_size)
        self._log_std = torch.nn.Linear(hid_size, latent_size)

    def __call__(self, x):
        # (batch_size, w, d) -> (batch_size, w*d)
        x = x.view(x.shape[0], -1)
        h0 = self._hid(x)

        mean = self._mean(h0)
        sigma = torch.exp(self._log_std(h0))
        return mean, sigma


class Decoder(torch.nn.Module):
    def __init__(self, output_shape, hid_size):
        super().__init__()
        self._output_shape = output_shape
        flat_features = reduce(mul, output_shape, 1)
        self._layer = torch.nn.Sequential(
            torch.nn.Linear(hid_size, hid_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_size, flat_features),
        )

    def __call__(self, z):
        x = self._layer(z)
        x_reconstructed = x.view(-1, *self._output_shape)
        return x_reconstructed


class VAE(torch.nn.Module):
    def __init__(self, image_shape, hid_size):
        super().__init__()
        self._encoder = Encoder(image_shape, latent_size=hid_size)
        self._decoder = Decoder(image_shape, hid_size=hid_size)

    def forward(self, x):
        z, (mean, stddev) = self.encode(x)
        logits, mean_image, sampled_image = self.decode(z)
        return mean_image, sampled_image, logits, z, mean, stddev

    def encode(self, x):
        mean, stddev = self._encoder(x)
        z = torch.normal(mean, stddev)
        return z, (mean, stddev)

    def decode(self, z):
        logits = self._decoder(z)
        mean_image = torch.nn.functional.sigmoid(logits)
        sampled_image = torch.bernoulli(mean_image)
        return logits, mean_image, sampled_image


def kl_gaussian(mean, var):
    kl_divergence_vector = 0.5 * (-torch.log(var) - 1.0 + var + mean**2)
    return torch.sum(kl_divergence_vector, axis=-1)


# For some reason num_workers doesn't work with skorch :/
class DataIterator(torch.utils.data.DataLoader):
    def __init__(self, dataset, num_workers=4, *args, **kwargs):
        super().__init__(dataset, num_workers=num_workers, *args, **kwargs)


class ShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        x, y = next(iter(X))
        net.set_params(module__image_shape=x.shape)
        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model():
    model = skorch.NeuralNet(
        module=VAE,
        module__image_shape=(2, 10, 10),
        module__hid_size=2,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        # criterion=MSE,
        max_epochs=20,
        batch_size=64,
        iterator_train=DataIterator,
        iterator_train__shuffle=True,
        # iterator_tarin__num_workers=2,
        iterator_valid=DataIterator,
        iterator_valid__shuffle=False,
        # iterator_valid__num_workers=2,
        train_split=None,
        callbacks=[
            ShapeSetter(),
        ]
    )
    return model


def visualize(encode, dataset, class_labels):
    zs, ys = [], []
    for (x, labels) in dataset:
        ys.append(labels)
        zs.append(encode(x).detach().numpy())

    latents = np.concatenate(zs, axis=0)
    labels = np.concatenate(ys, axis=0)

    for yy in range(10):
        plt.scatter(
            latents[labels == yy][:, 0],
            latents[labels == yy][:, 1],
            label=class_labels[yy], s=6)
    plt.legend()
    plt.scatter([0], [0], marker="x", s=150, color="black", linewidth=3)
    plt.axis('equal')


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])

    train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform,
    )

    class_labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
        # 'zero', 'one', 'two', 'three', 'four', 'five', 'six',
        # 'seven', 'eight', 'nine'
    ]

    model = build_model().fit(train)
    encode = model.module_.encode

    visualize(encode, DataIterator(train, batch_size=4), class_labels)
    plt.plot()
    plt.show()


if __name__ == '__main__':
    main()
