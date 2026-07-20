"""CNN classifier/regressor for downstream evaluation."""

import torch
import torch.nn as nn


class CNN_classifier(nn.Module):
    """Simple 1-D convolutional classifier/regressor for raw time-series inputs."""

    def __init__(
        self,
        input_dim,
        out_channels,
        kernel_size,
        output_dim,
        dense_dim,
        input_length,
        regression=False,
    ):
        super().__init__()
        self.regression = regression
        channels = [input_dim] + list(out_channels)
        layers = []
        for in_channels, next_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        next_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, input_length)
            flattened_dim = self.features(dummy).reshape(1, -1).shape[1]

        self.class_fc = nn.Sequential(
            nn.Linear(flattened_dim, dense_dim),
            nn.GELU(),
            nn.Linear(dense_dim, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != self.features[0].in_channels:
            x = x.transpose(1, 2)
        output = self.class_fc(self.features(x).reshape(x.size(0), -1))
        if not self.regression and output.shape[-1] == 1:
            return output.squeeze(-1)
        return output
