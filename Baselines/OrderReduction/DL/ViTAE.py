"""CAE with attention between the convolutions and the latent (a hybrid ViT autoencoder).

Encoder: the CAE convolutions give a (channels, height, width) feature map. Each pixel of that
map is one token of size channels, projected to hidden. A learned summary token (CLS) is
prepended and a learned position embedding is added, then attention layers mix the tokens. A
final feed-forward maps the CLS token to the latent.

Decoder: a linear layer maps the latent to one token, which is copied to every position and
added to a learned position embedding. After the attention layers a final feed-forward maps
each token back to channels, the tokens are reshaped into the feature map, and the CAE
transposed convolutions rebuild the image.
"""
import torch
from torch import nn
from .CAE import CAE


def attention_layers(hidden, heads, layers, feedforward):
    layer = nn.TransformerEncoderLayer(hidden, heads, feedforward, dropout=0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TokenEncoder(nn.Module):
    def __init__(self, input_shape_chw, output_dim, hidden, heads, layers, feedforward):
        super().__init__()
        channels, height, width = input_shape_chw
        self.embedding = nn.Linear(channels, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position = nn.Parameter(torch.randn(1, height * width + 1, hidden) * 0.02)
        self.attention = attention_layers(hidden, heads, layers, feedforward)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, output_dim))

    def forward(self, x):
        tokens = self.embedding(x.flatten(2).transpose(1, 2))  # (batch, height * width, hidden)
        tokens = torch.cat((self.cls.expand(len(x), -1, -1), tokens), dim=1) + self.position
        return self.head(self.attention(tokens)[:, 0])


class TokenDecoder(nn.Module):
    def __init__(self, input_shape_chw, rank, hidden, heads, layers, feedforward):
        super().__init__()
        channels, height, width = self.input_shape_chw = input_shape_chw
        self.embedding = nn.Linear(rank, hidden)
        self.position = nn.Parameter(torch.randn(1, height * width, hidden) * 0.02)
        self.attention = attention_layers(hidden, heads, layers, feedforward)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, channels))

    def forward(self, z):
        tokens = self.attention(self.embedding(z)[:, None] + self.position)  # (batch, height * width, hidden)
        return self.head(tokens).transpose(1, 2).reshape(len(z), *self.input_shape_chw)


class ViTAE(CAE):
    """hidden: token size; heads: attention heads (must divide hidden); layers: attention layers
    in both encoder and decoder; feedforward: MLP size inside each layer (default 4 * hidden).
    The convolution keys (channels, kernel_size, stride, padding) are those of CAE."""
    name = "vit_ae"

    def __init__(self, hidden=64, heads=4, layers=2, feedforward=None, **kwargs):
        super().__init__(**kwargs)
        if min(hidden, heads, layers) < 1 or hidden % heads:
            raise ValueError("Positive dimensions and hidden divisible by heads required")
        self.hidden, self.heads, self.layers = hidden, heads, layers
        self.feedforward = feedforward or 4 * hidden

    def build(self):
        encoder, decoder, input_shape_chw = self.convolutions()
        attention = dict(hidden=self.hidden, heads=self.heads, layers=self.layers, feedforward=self.feedforward)
        self.encoder = nn.Sequential(*encoder, TokenEncoder(input_shape_chw, self.encoder_outputs, **attention))
        self.decoder = nn.Sequential(TokenDecoder(input_shape_chw, self.rank, **attention), *decoder)
