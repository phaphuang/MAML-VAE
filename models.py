from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from typing import Dict

##########
# Layers #
##########
class Flatten(nn.Module):
    """ Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class GlobalMaxPool1d(nn.Module):
    """ Performs global max pooling over the entire length of a batched 1D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))

class GlobalAvgPool2d(nn.Module):
    """ Performs global average pooling over the entire height and width of a batched 2D tensor
    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))

def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """ Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 maxpooling.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor, bn_weights, bn_biases) -> torch.Tensor:
    """ Performs 3x3 convolution, ReLU activation, 2x2 max pooling in a functional fashion.

    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x

class FewShotClassifier(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int = 64):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(FewShotClassifier, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        self.logits = nn.Linear(final_layer_size, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""

        for block in [1, 2, 3, 4]:
            x = functional_conv_block(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                      weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x
    
# src: https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
class VAE(nn.Module):
    def __init__(self, in_channels: int, input_size: int, z_dim: int=16):
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            Flatten()
        )

        self.fc1 = nn.Linear(7 * 7 * 32, z_dim)
        self.fc2 = nn.Linear(7 * 7 * 32, z_dim)

        self.fc3 = nn.Linear(z_dim, 7 * 7 * 32)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(7 * 7 * 32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar
    
    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

