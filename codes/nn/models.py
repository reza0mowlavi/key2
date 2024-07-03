from dataclasses import dataclass
from typing import Sequence
from itertools import chain

import torch
from torch import nn

from .activations import ACT2FN
from .layers import RMSNorm, GlobalAvgPooling1D, PrependCLSToken
from .transformer import (
    SinusoidalEmbedding,
    LearnableEmbedding,
    Encoder,
    CrossAtentionLayer,
    create_attention_mask,
    create_sequence_mask,
    create_look_ahead_mask,
)


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


def input_length_after_conv(input_length, kernel_size, stride, padding=None):
    if padding is None:
        padding = [0] * len(kernel_size)

    for ks, s, p in zip(kernel_size, stride, padding):
        input_length = _input_length_after_a_conv(
            input_length=input_length,
            kernel_size=ks,
            stride=s,
            padding=p,
        )
    return input_length


def _input_length_after_a_conv(input_length, kernel_size, stride, padding):
    if padding == "same":
        return torch.ceil(input_length / stride)

    return (
        torch.div(
            torch.subtract(input_length, kernel_size) + 2 * padding,
            stride,
            rounding_mode="floor",
        )
        + 1
    )


class _Conv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        stride=1,
        bias=True,
        channels_last=True,
        norm=None,
        norm_eps=1e-6,
        conv_kwds=None,
    ):
        super().__init__()
        conv_kwds = {} if conv_kwds is None else conv_kwds
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **conv_kwds,
        )

        self.activation = (
            ACT2FN[activation] if isinstance(activation, str) else activation
        )

        if norm is None:
            self.norm = None
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(out_channels, eps=norm_eps)
        elif norm == "rmsnorm":
            self.norm = RMSNorm(out_channels, eps=norm_eps)
        else:
            raise ValueError(f'norm "{norm}" is not supported.')

        self.forward = (
            self.forward_channel_last if channels_last else self.forward_channel_first
        )

    def forward_channel_last(self, x):
        x = x.transpose(-2, -1)
        x = self.conv(x)
        x = x.transpose(-2, -1)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        return x

    def forward_channel_first(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = x.transpose(-2, -1)
            x = self.norm(x)
            x = x.transpose(-2, -1)
        x = self.activation(x)
        return x


class Conv1D(nn.Module):
    def __init__(
        self,
        input_dim,
        out_channels,
        kernel_size,
        stride,
        activation="gelu",
        bias=True,
        norm=None,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.channels_last = False if norm is None else True
        conv_layers = [
            _Conv1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                stride=stride,
                bias=bias,
                channels_last=self.channels_last,
                norm=norm,
                norm_eps=norm_eps,
            )
            for in_channels, out_channels, kernel_size, stride in zip(
                chain([input_dim], out_channels[:-1]),
                out_channels,
                kernel_size,
                stride,
            )
        ]
        self.conv_layers = nn.Sequential(conv_layers)

    def forward(self, x):
        if not self.channels_last:
            x = x.transpose(-2, -1)

        for layer in self.conv_layers:
            x = layer(x)

        if not self.channels_last:
            x = x.transpose(-2, -1)

        return x


class Projection(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        dropout=0.0,
        norm_eps=1e-6,
        norm=False,
        use_rms_norm=False,
    ):
        super().__init__()

        self.project = nn.Linear(input_dim, hidden_size)
        self.norm = None
        if norm:
            self.norm = (
                RMSNorm(hidden_size, eps=norm_eps)
                if use_rms_norm
                else nn.LayerNorm(hidden_size, eps=norm_eps)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.project(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.dropout(x)
        return x


@dataclass
class TransformerConfig:
    input_dim: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    mha_type: str
    hidden_dropout: float
    attention_dropout: float
    dropout_feedforward: float
    project_dropout: float
    dropout_clf: float
    kernel_size: Sequence[int] = None
    stride: Sequence[int] = None
    out_channels: Sequence[int] = None
    activation_feedforward: str = "gelu"
    dim_feedforward: int = None
    bias: bool = True
    maxlen: int = None
    prepend_cls_token: bool = False
    use_positional_embedding: bool = False
    sinusoidal: bool = True
    rotary_base: int = 10_000
    rope_percentage: float = 0.5
    norm_eps: float = 1e-8
    conv_norm: bool = None
    project_norm: bool = True
    use_rms_norm: bool = False
    layer_dropout: float = False
    causal_attention: bool = False

    def __post_init__(self):
        if self.causal_attention and self.prepend_cls_token:
            raise Exception(
                'It is contradictory to have both "prepend_cls_token" and "causal_attention" on.'
            )


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.conv = None
        if config.kernel_size is not None:
            self.conv = Conv1D(
                input_dim=config.input_dim,
                activation=config.activation_feedforward,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                bias=config.bias,
                norm=config.conv_norm,
                norm_eps=config.norm_eps,
            )

        projection_input_dim = (
            config.input_dim if config.kernel_size is None else config.out_channels[-1]
        )
        self.projection = Projection(
            input_dim=projection_input_dim,
            hidden_size=config.hidden_size,
            norm_eps=config.norm_eps,
            dropout=config.project_dropout,
            use_rms_norm=config.use_rms_norm,
            norm=config.project_norm,
        )

        self.prepend_cls_token = self.global_pool = None
        if config.prepend_cls_token:
            self.prepend_cls_token = PrependCLSToken(hidden_size=config.hidden_size)
            self.cross_attn_layer = CrossAtentionLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                hidden_dropout=config.hidden_dropout,
                attention_dropout=config.attention_dropout,
                activation_feedforward=config.activation_feedforward,
                dim_feedforward=config.dim_feedforward,
                dropout_feedforward=config.dropout_feedforward,
                bias=config.bias,
                mha_type=config.mha_type,
                rotary_base=config.rotary_base,
                rope_percentage=config.rope_percentage,
                norm_eps=config.norm_eps,
                use_rms_norm=config.use_rms_norm,
            )
        else:
            self.global_pool = GlobalAvgPooling1D()

        self.pe_embeddings = None
        if config.use_positional_embedding:
            if config.sinusoidal:
                self.pe_embeddings = SinusoidalEmbedding(config.hidden_size)
            else:
                if config.maxlen is None:
                    raise ValueError(
                        "When using learnable positional embedding, 'maxlen' must be set."
                    )
                self.pe_embeddings = LearnableEmbedding(
                    hidden_size=config.hidden_size, maxlen=config.maxlen
                )

        self.encoder = Encoder(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_dropout=config.hidden_dropout,
            attention_dropout=config.attention_dropout,
            activation_feedforward=config.activation_feedforward,
            dim_feedforward=config.dim_feedforward,
            dropout_feedforward=config.dropout_feedforward,
            bias=config.bias,
            layer_dropout=config.layer_dropout,
            mha_type=config.mha_type,
            rotary_base=config.rotary_base,
            rope_percentage=config.rope_percentage,
            norm_eps=config.norm_eps,
            use_rms_norm=config.use_rms_norm,
        )

        self.clf = nn.Sequential(
            nn.Dropout(config.dropout_clf), nn.Linear(config.hidden_size, 1)
        )

    def _create_attention_mask(self, input_length):
        sequence_mask = create_sequence_mask(input_length)  ## (BS, T_max)
        enc_attn_mask = create_attention_mask(sequence_mask)  ## (BS, T_max,T_max)

        if self.config.causal_attention:
            enc_attn_mask = create_look_ahead_mask(enc_attn_mask)

        cross_attn_mask = None
        if self.prepend_cls_token is not None:
            cross_attn_mask = create_attention_mask(
                sequence_mask[:, 1:],  ## Due to removal of cls token
                dec_mask=torch.ones(
                    input_length.size(0),
                    1,
                    dtype=torch.bool,
                    device=input_length.device,
                ),
            )

        return {
            "enc_attn_mask": enc_attn_mask,
            "sequence_mask": sequence_mask,
            "cross_attn_mask": cross_attn_mask,
        }

    def forward(self, x, input_length=None):
        if self.conv is not None:
            x = self.conv(x)
            if input_length is not None:
                input_length = input_length_after_conv(
                    input_length=input_length,
                    kernel_size=self.config.kernel_size,
                    stride=self.config.stride,
                )

        x = self.projection(x)  ### (BS, S, input_dim) -> (BS, S, d)

        if self.prepend_cls_token is not None:
            x = self.prepend_cls_token(x)
            if input_length is not None:
                input_length += 1

        enc_attn_mask = mask_info = None
        if input_length is not None:
            mask_info = self._create_attention_mask(input_length)
            enc_attn_mask = mask_info["enc_attn_mask"]

        x = self.encoder(x, attention_mask=enc_attn_mask)  ### (BS, S, d)
        x = (
            self.cross_attn_layer_forward(x, mask_info)
            if self.prepend_cls_token is not None
            else self.global_pool_forward(x, mask_info)
        )
        x = self.clf(x)
        return x

    def cross_attn_layer_forward(self, x, mask_info):
        cls_token = x[:, 0:1, :]  ### (BS, 1, d)
        memory = x[:, 1:, :]  ### (BS, S, d)
        cross_attn_mask = None if mask_info is None else mask_info["cross_attn_mask"]
        x = self.cross_attn_layer(
            x=cls_token, memory=memory, attention_mask=cross_attn_mask
        )  ### (BS, 1, d)
        x = x.squeeze(dim=1)  ### (BS, d)
        return x

    def global_pool_forward(self, x, mask_info):
        sequence_mask = None if mask_info is None else mask_info["sequence_mask"]
        x = self.global_pool(x, sequence_mask)
        return x
