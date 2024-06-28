from .activations import ACT2FN

from torch import nn


def make_linear(in_dim, out_dim, dropout_rate, activation, layer_normalization):
    layers = []
    layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))

    if layer_normalization:
        layers.append(nn.LayerNorm(out_dim))

    if activation is not None:
        layers.append(ACT2FN[activation] if isinstance(activation, str) else activation)

    if dropout_rate is not None:
        layers.append(nn.Dropout(dropout_rate))

    return layers


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_dims,
        output_dim,
        dropout_rates=None,
        output_dropout_rate=None,
        layer_normalization=False,
        activation="gelu",
        output_activation=None,
    ):
        super().__init__()
        dropout_rates = (
            [None] * (1 + len(hidden_layer_dims))
            if dropout_rates is None
            else dropout_rates
        )

        layers = []
        for in_dim, out_dim, dropout_rate in zip(
            [input_dim] + list(hidden_layer_dims[:-1]),  ### in_dims
            list(hidden_layer_dims),  ### out_dims
            dropout_rates,
        ):
            layers.extend(
                make_linear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    layer_normalization=layer_normalization,
                )
            )

        layers.extend(
            make_linear(
                in_dim=hidden_layer_dims[-1],
                out_dim=output_dim,
                dropout_rate=output_dropout_rate,
                layer_normalization=layer_normalization,
                activation=output_activation,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
