"""Transformer-based classification/regression heads for downstream evaluation."""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learned attention pooling over encoded time-series patches."""

    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class Transformer_based(nn.Module):
    """Classifier/regressor that reuses an Encoder and adds a dense prediction head."""

    def __init__(
        self,
        encoder,
        embed_dim,
        dense_dim,
        patch_size,
        num_patch,
        n_classes,
        pooling="Mean",
        regression=False,
        pretrained=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patch = num_patch
        self.n_classes = n_classes
        self.pooling = pooling
        self.regression = regression
        self.pretrained = pretrained

        if str(pooling).lower() == "attention":
            self.attention_pooling = AttentionPooling(embed_dim)
            head_input_dim = embed_dim
        elif str(pooling).lower() == "flatten":
            self.attention_pooling = None
            head_input_dim = embed_dim * num_patch
        else:
            self.attention_pooling = None
            head_input_dim = embed_dim

        self.class_fc = nn.Sequential(
            nn.Linear(head_input_dim, dense_dim),
            nn.GELU(),
            nn.Linear(dense_dim, n_classes),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        pooling = str(self.pooling).lower()
        if pooling == "attention":
            features = self.attention_pooling(encoded)
        elif pooling == "flatten":
            features = encoded.reshape(encoded.size(0), -1)
        else:
            features = encoded.mean(dim=1)

        output = self.class_fc(features)
        if not self.regression and output.shape[-1] == 1:
            return output.squeeze(-1)
        return output
