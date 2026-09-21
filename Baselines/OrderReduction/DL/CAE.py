"""Convolutional image autoencoder whose decoder mirrors the encoder.

Each encoder level is Conv2d(kernel_size, stride, padding) + SiLU. The decoder repeats the levels
in reverse with ConvTranspose2d and the same kernel_size, stride and padding. A convolution
rounds down, out = (size + 2 * padding - kernel_size) // stride + 1, so each transposed
convolution gets output_padding = the remainder of that division, which restores the exact
input size. A linear layer maps the last feature map to the latent and back.
"""
from torch import nn
from .AE import AE


class CAE(AE):
    name = "cae"

    def __init__(self, channels=(16, 32, 64), kernel_size=3, stride=2, padding=1, **kwargs):
        super().__init__(**kwargs)
        if not channels or min(channels) < 1:
            raise ValueError("channels must be a nonempty list of positive widths, one per level")
        self.channels = tuple(channels)
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def convolutions(self):
        """Encoder Conv2d levels, mirrored decoder ConvTranspose2d levels, last feature map shape."""
        fields, height, width = self.shape
        widths = (fields, *self.channels)
        encoder, decoder = [], []
        for before, after in zip(widths[:-1], widths[1:]):
            height, height_remainder = divmod(height + 2 * self.padding - self.kernel_size, self.stride)
            width, width_remainder = divmod(width + 2 * self.padding - self.kernel_size, self.stride)
            height, width = height + 1, width + 1
            if min(height, width) < 1:
                raise ValueError(f"Image {self.shape[1:]} too small for {len(self.channels)} levels")
            encoder += [nn.Conv2d(before, after, self.kernel_size, self.stride, self.padding), nn.SiLU()]
            decoder = [nn.ConvTranspose2d(after, before, self.kernel_size, self.stride, self.padding,
                                          output_padding=(height_remainder, width_remainder)), nn.SiLU()] + decoder
        return encoder, decoder[:-1], (self.channels[-1], height, width)  # No activation on the output.

    def build(self):
        encoder, decoder, input_shape_chw = self.convolutions()
        size = input_shape_chw[0] * input_shape_chw[1] * input_shape_chw[2]
        self.encoder = nn.Sequential(*encoder, nn.Flatten(), nn.Linear(size, self.encoder_outputs))
        self.decoder = nn.Sequential(nn.Linear(self.rank, size), nn.Unflatten(1, input_shape_chw), *decoder)
