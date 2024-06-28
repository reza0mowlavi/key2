from .activations import softmax, ACT2FN
from .layers import RMSNorm

import torch
from torch import nn


def create_indices(sequence_len, device):
    rows = torch.arange(sequence_len, device=device).view(-1, 1).repeat(1, sequence_len)

    cols = (
        torch.arange(sequence_len, device=device)
        .view(1, sequence_len)
        .repeat((sequence_len, 1))
        - torch.arange(sequence_len, device=device).view(-1, 1) % sequence_len
    ).abs_()
    return rows, cols


def create_negative_lower(sequence_len, dtype, device):
    ones = torch.ones(sequence_len, sequence_len, dtype=dtype, device=device)
    return torch.triu(ones) - torch.tril(ones, diagonal=-1)


def create_sequence_mask(input_lengths, maxlen=None):
    maxlen = maxlen if maxlen is not None else torch.max(input_lengths)
    shape = (input_lengths.size(0), int(maxlen))
    return ~(
        torch.ones(*shape, dtype=input_lengths.dtype, device=input_lengths.device)
        .cumsum(axis=1)
        .T
        > input_lengths
    ).T


def create_attention_mask(mask, dec_mask=None):
    dec_mask = dec_mask if dec_mask is not None else mask
    attention_mask = torch.einsum("bi,bj->bij", dec_mask, mask)  ## (BS, T, S)
    return attention_mask


def create_look_ahead_mask(attention_mask, dtype=None, device=None, size=None):
    size = size if size is not None else attention_mask.size(-1)
    dtype = dtype if dtype is not None else attention_mask.dtype
    device = device if device is not None else attention_mask.device

    mask = torch.tril(torch.ones(size, size, dtype=dtype, device=device))
    if attention_mask is not None:
        mask = torch.einsum("bij,ij->bij", attention_mask, mask)
    return mask


class SinusoidalEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        inv_freq = 1 / (
            10000
            ** (torch.arange(0.0, hidden_size, 2.0, dtype=torch.float32) / hidden_size)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, key, key_len=None, device=None):
        key_len = key.size(1) if key_len is None else key_len
        device = key.device if device is None else device
        pos_seq = torch.arange(key_len, dtype=torch.float32, device=device)
        sinusoid_inp = torch.einsum("i,j->ij", pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class LearnableEmbedding(nn.Module):
    def __init__(self, hidden_size, maxlen):
        super().__init__()
        self.embeddings = nn.Parameter(data=torch.empty(maxlen, hidden_size))
        nn.init.normal_(self.embeddings)

    def forward(self, x):
        seq_len = x.size(1)
        return self.embeddings[:seq_len]


class PositionwiseFF(nn.Module):
    def __init__(
        self,
        hidden_size,
        dim_feedforward=None,
        activation_feedforward=nn.functional.relu,
        dropout_feedforward=0.0,
        bias=True,
    ):
        super().__init__()
        dim_feedforward = (
            4 * hidden_size if dim_feedforward is None else dim_feedforward
        )

        self.intermediate_dropout = nn.Dropout(dropout_feedforward)

        self.intermediate_dense = nn.Linear(hidden_size, dim_feedforward, bias=bias)
        self.intermediate_act_fn = (
            ACT2FN[activation_feedforward]
            if isinstance(activation_feedforward, str)
            else activation_feedforward
        )

        self.output_dense = nn.Linear(dim_feedforward, hidden_size, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.scale = 1.0 / float(key_dim) ** 0.5
        self.dropout = dropout

        self.q_net = nn.Linear(
            hidden_size,
            num_attention_heads * self.key_dim,
            bias=bias,
        )
        self.k_net = nn.Linear(
            hidden_size,
            num_attention_heads * self.key_dim,
            bias=bias,
        )
        self.v_net = nn.Linear(
            hidden_size,
            num_attention_heads * self.value_dim,
            bias=bias,
        )

        self.o_net = nn.Linear(
            num_attention_heads * self.value_dim,
            hidden_size,
            bias=bias,
        )

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        ### query => (BS,T,num_heads,key_dim)
        ### key => (BS,S,num_heads,key_dim)
        return torch.einsum("btnd,bsnd->tsbn", query, key)  ## (T,S,BS,num_heads)

    def scaled_dot_product_attention(self, query, key, value, attention_mask=None):
        ### query => (BS,T,num_heads,key_dim)
        ### key => (BS,S,num_heads,key_dim)
        ### value => (BS,S,num_heads,val_dim)
        attn_score = self.get_scores(query=query, key=key)  ## (T,S,BS,num_heads)
        attn_score = attn_score * self.scale  ## (T,S,BS,num_heads)

        # compute attention probability

        if attention_mask is not None:
            attention_mask = attention_mask.permute(1, 2, 0)  ## (BS,T,S) -> (T, S, BS)
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)  ## (T, S, BS, 1)

        attn_score = softmax(
            attn_score, mask=attention_mask, dim=1
        )  ## (T,S,BS,num_heads)

        attn_score = nn.functional.dropout(  ## (T,S,BS,num_heads)
            input=attn_score, p=self.dropout, training=self.training
        )
        attn_output = torch.einsum(
            "ijbn,bjnd->bind", attn_score, value
        )  ## (BS,T,num_heads,value_dim)

        return attn_output

    def forward(self, query, value, key=None, attention_mask=None):
        ## positional_embedding in (S, hidden_size)
        key = key if key is not None else value
        value_dim = self.value_dim

        batch_size = query.size(0)

        query = self.q_net(query).view(
            batch_size, -1, self.num_attention_heads, self.key_dim
        )  ## (BS,T,num_heads,key_dim)
        key = self.k_net(key).view(
            batch_size, -1, self.num_attention_heads, self.key_dim
        )  ## (BS,S,num_heads,key_dim)
        value = self.v_net(value).view(
            batch_size, -1, self.num_attention_heads, value_dim
        )  ## (BS,S,num_heads,value_dim)

        attn_output = self.scaled_dot_product_attention(
            query=query, key=key, value=value, attention_mask=attention_mask
        )  ## (BS,T,num_heads,value_dim)
        attn_output = attn_output.reshape(
            attn_output.size(0),
            attn_output.size(1),
            -1,
        )  ## (BS,T,num_heads,value_dim) -> (BS,T,num_heads*value_dim)

        final_output = self.o_net(attn_output)  ## (BS,T,d_model)

        return final_output


class NativeMultiHeadAttention(MultiHeadAttention):
    def scaled_dot_product_attention(self, query, key, value, attention_mask=None):
        ### query => (BS,T,num_heads,key_dim)
        ### key => (BS,S,num_heads,key_dim)
        ### value => (BS,S,num_heads,val_dim)
        ### attention_mask => (BS,T,S)
        query = query.permute(
            2, 0, 1, 3
        )  ## (BS,T,num_heads,key_dim) -> (num_heads,BS,T,key_dim)

        value = value.permute(
            2, 0, 1, 3
        )  ## (BS,T,num_heads,key_dim) -> (num_heads,BS,T,key_dim)

        key = key.permute(
            2, 0, 1, 3
        )  ## (BS,T,num_heads,value_dim) -> (num_heads,BS,T,value_dim)

        dropout = self.dropout if self.training else 0.0

        attn_output = nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=self.scale,
        )  ## (num_heads,BS,T,val_dim)

        if attention_mask is not None:
            attn_output = torch.nan_to_num(attn_output, nan=0.0)

        attn_output = attn_output.permute(
            1, 2, 0, 3
        )  ## (num_heads,BS,T,value_dim) -> (BS,T,num_heads,value_dim)
        return attn_output


### Taken from https://github.com/labmlai/annotated_deep_learning_paper_implementations
class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.

    ### For a pair of features

    Let $x^{(1)}_m$ and $x^{(2)}_m$ be two features of the
    key or query of any head at position $m$.
    Or for simplicity assume $x$ has only two features.
    Then the transformation is,

    \begin{align}
    RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big) &=
    \begin{pmatrix}
    \cos m \theta & - \sin m \theta \\
    \sin m \theta & \cos m \theta
    \end{pmatrix}
    \begin{pmatrix}
    x^{(1)}_m \\
    x^{(2)}_m \\
    \end{pmatrix} \\
    &=
    \begin{pmatrix}
    x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta \\
    x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta \\
    \end{pmatrix} \\
    \end{align}

    where $\theta$ is a constant angle. The other pairs of features are transformed similarly.

    ### Attention is relative

    For a pair of features, dot-product attention score between two positions $m$ and $n$ would be

    \begin{align}
    \Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, n\big) \Big \rangle &= \\
    (x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta)(x^{(1)}_n \cos n\theta - x^{(2)}_n \sin n \theta) &+ \\
    (x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta)(x^{(2)}_n \cos n\theta + x^{(1)}_n \sin n \theta) &= \\
    x^{(1)}_m x^{(1)}_n (\cos m\theta \cos n\theta + \sin m \theta \sin n \theta) &+ \\
    x^{(1)}_m x^{(2)}_n (-\cos m\theta \sin n\theta + \sin m \theta \cos n \theta) &+ \\
    x^{(2)}_m x^{(1)}_n (-\sin m\theta \cos n\theta + \cos m \theta \sin n \theta) &+ \\
    x^{(2)}_m x^{(2)}_n (\sin m\theta \sin n\theta + \cos m \theta \cos n \theta) &= \\

    x^{(1)}_m x^{(1)}_n \cos (m - n) \theta +
    x^{(1)}_m x^{(2)}_n \sin(m - n) \theta &+ \\
    - x^{(2)}_m x^{(1)}_n \sin (m - n) \theta +
    x^{(2)}_m x^{(1)}_n \cos (m - n) \theta &= \\

    \big(x^{(1)}_m \cos (m - n)\theta - x^{(2)}_m \sin (m - n) \theta\big) x^{(1)}_n &+ \\
    \big(x^{(2)}_m \cos (m - n)m\theta + x^{(1)}_m \sin (m - n) \theta\big) x^{(2)}_n  &= \\

    \Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m - n\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, 0\big) \Big \rangle
    \end{align}

    This shows that for dot-production attention the rotary encodings gives relative attention.

    ### For all features

    The features are grouped into pairs and handled as above. They use a different $\theta$ for each pair.

    The paper suggests using $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    for the $\frac{d}{2}$ pairs of features.

    We pair feature $i$ with feature $i + \frac{d}{2}$. So for position $m$ we transform

    \begin{align}
    \begin{pmatrix}
    x^{(i)}_m \\
    x^{(i + \frac{d}{2})}_m
    \end{pmatrix}
    \end{align}

    to

    \begin{align}
    \begin{pmatrix}
    x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
    x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
    \end{pmatrix} \\
    \end{align}
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None
        self.cos_cached_in_infer = None
        self.sin_cached_in_infer = None

    @torch.no_grad()
    def _build_cache(self, x: torch.Tensor):
        """
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built

        if torch.is_inference_mode_enabled():
            if (
                self.cos_cached_in_infer is not None
                and x.shape[0] <= self.cos_cached_in_infer.shape[0]
            ):
                return
        else:
            if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
                return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.d, 2, dtype=x.dtype, device=x.device) / self.d)
        )

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device, dtype=x.dtype)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        if torch.is_inference_mode_enabled():
            self.cos_cached_in_infer = idx_theta2.cos()[:, None, None, :]
            self.sin_cached_in_infer = idx_theta2.sin()[:, None, None, :]
        else:
            self.cos_cached = idx_theta2.cos()[:, None, None, :]
            self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        cos_cached = (
            self.cos_cached_in_infer
            if torch.is_inference_mode_enabled()
            else self.cos_cached
        )
        sin_cached = (
            self.sin_cached_in_infer
            if torch.is_inference_mode_enabled()
            else self.sin_cached
        )
        x_rope = (x_rope * cos_cached[: x.shape[0]]) + (
            neg_half_x * sin_cached[: x.shape[0]]
        )

        #
        return torch.cat((x_rope, x_pass), dim=-1)


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    """
    ## Multi-head attention with rotary positional embeddings

    We override [multi-head attention from original transformer](../mha.html).
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        key_dim,
        rope_percentage=0.5,
        base=10_000,
        value_dim=None,
        dropout=0.0,
        bias=True,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dropout=dropout,
            bias=bias,
        )

        # Rotary positional embedding layers
        d_rope = int(key_dim * rope_percentage)
        if not d_rope % 2 == 0:
            d_rope += 1

        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope, base=base)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope, base=base)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        """
        ### query => (BS,T,num_heads,key_dim)
        ### key => (BS,S,num_heads,key_dim)
        query = torch.einsum(
            "bihd->ibhd", query
        )  ### (BS,T,num_heads,key_dim) => (T,BS,num_heads,key_dim)
        key = torch.einsum(
            "bjhd->jbhd", key
        )  ### (BS,S,num_heads,key_dim) => (S,BS,num_heads,key_dim)
        # Calculate dot-product with RoPE
        return torch.einsum(
            "ibhd,jbhd->ijbh",
            self.query_rotary_pe(query),  ## (T,BS,num_heads,key_dim)
            self.key_rotary_pe(key),  ## (S,BS,num_heads,key_dim)
        )


class XLMultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.scale = 1.0 / float(key_dim) ** 0.5

        self.q_net = nn.Linear(
            hidden_size,
            num_attention_heads * self.key_dim,
            bias=bias,
        )
        self.k_net = nn.Linear(
            hidden_size,
            num_attention_heads * self.key_dim,
            bias=bias,
        )
        self.v_net = nn.Linear(
            hidden_size,
            num_attention_heads * self.value_dim,
            bias=bias,
        )

        self.r_net = nn.Linear(
            hidden_size,
            self.num_attention_heads * self.key_dim,
            bias=bias,
        )

        self.o_net = nn.Linear(
            num_attention_heads * self.value_dim,
            hidden_size,
            bias=bias,
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        value,
        positional_embedding,
        u_weight,
        v_weight,
        indices,
        lower,
        key=None,
        attention_mask=None,
    ):
        ## positional_embedding in (S, hidden_size)
        key = key if key is not None else value
        value_dim = self.value_dim

        batch_size = query.size(0)
        relative_len = positional_embedding.size(0)

        query = self.q_net(query).view(
            batch_size, -1, self.num_attention_heads, self.key_dim
        )  ## (BS,T,num_heads,key_dim)
        key = self.k_net(key).view(
            batch_size, -1, self.num_attention_heads, self.key_dim
        )  ## (BS,S,num_heads,key_dim)
        value = self.v_net(value).view(
            batch_size, -1, self.num_attention_heads, value_dim
        )  ## (BS,S,num_heads,value_dim)

        positional_embedding = self.r_net(positional_embedding).view(
            relative_len, self.num_attention_heads, self.key_dim
        )  ## (S,num_heads,key_dim)

        q_u = query + u_weight  ## (BS,T,num_heads,key_dim)
        q_v = query + v_weight  ## (BS,T,num_heads,key_dim)

        AC = torch.einsum("bind,bjnd->ijbn", q_u, key)  ## (T,S,BS,num_heads)
        BD = torch.einsum(
            "bind,jnd->ijbn", q_v, positional_embedding
        )  ## (T,S,BS,num_heads)
        BD = self._rel_shift(
            BD,
            indices=indices,
            lower=lower,
        )  ## (T,S,BS,num_heads)

        attn_score = AC + BD  ## (T,S,BS,num_heads)
        attn_score = attn_score * self.scale

        # compute attention probability

        if attention_mask is not None:
            attention_mask = attention_mask.permute(1, 2, 0)  ## (T, S, BS)
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)  ## (T, S, BS, 1)

        attn_score = softmax(attn_score, dim=1, mask=attention_mask)

        attn_score = self.attn_dropout(attn_score)  ## (T,S,BS,num_heads)
        attn_output = torch.einsum(
            "ijbn,bjnd->bind", attn_score, value
        )  ## (BS,T,num_heads,value_dim)
        attn_output = attn_output.contiguous().view(
            attn_output.size(0),
            attn_output.size(1),
            -1,
        )  ## (BS,T,num_heads*value_dim)

        final_output = self.o_net(attn_output)  ## (BS,T,d_model)

        return final_output

    def _rel_shift(
        self,
        x,
        indices,
        lower,
    ):
        ### x in (T, S, BS, num_heads)
        x = x[indices]  ## (T, S, BS, num_heads)
        x = torch.einsum("ijbn,ij->ijbn", x, lower)  ## (T, S, BS, num_heads)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_feedforward=nn.functional.relu,
        dim_feedforward=None,
        dropout_feedforward=0.0,
        key_dim=None,
        bias=True,
        mha_type="torch",
        rotary_base=10_000,
        rope_percentage=0.5,
        norm_eps=1e-6,
        use_rms_norm=False,
    ):
        super().__init__()
        self.config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
            "activation_feedforward": activation_feedforward,
            "dim_feedforward": dim_feedforward,
            "dropout_feedforward": dropout_feedforward,
            "key_dim": key_dim,
            "bias": bias,
            "mha_type": mha_type,
            "rotary_base": rotary_base,
            "rope_percentage": rope_percentage,
            "norm_eps": norm_eps,
            "use_rms_norm": use_rms_norm,
        }
        key_dim = hidden_size // num_attention_heads if key_dim is None else key_dim

        self.forward = self._forward

        if mha_type == "native":
            self.mha = NativeMultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                dropout=attention_dropout,
                value_dim=None,
                bias=bias,
            )
        elif mha_type == "torch":
            self.mha = MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                dropout=attention_dropout,
                value_dim=None,
                bias=bias,
            )
        elif mha_type == "rotary":
            self.mha = RotaryPEMultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                dropout=attention_dropout,
                value_dim=None,
                bias=bias,
                base=rotary_base,
                rope_percentage=rope_percentage,
            )
        elif mha_type == "xl":
            self.mha = XLMultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                value_dim=None,
                dropout=attention_dropout,
                bias=bias,
            )
            self.forward = self.xl_forward
        else:
            raise ValueError(f"mha_type= '{mha_type}' is not knwon.")

        self.feedforward = PositionwiseFF(
            hidden_size=hidden_size,
            dim_feedforward=dim_feedforward,
            activation_feedforward=activation_feedforward,
            dropout_feedforward=dropout_feedforward,
            bias=bias,
        )
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.dropout2 = nn.Dropout(hidden_dropout)

        NormClass = RMSNorm if use_rms_norm else nn.LayerNorm

        self.norm1 = NormClass(
            hidden_size,
            eps=norm_eps,
        )
        self.norm2 = NormClass(
            hidden_size,
            eps=norm_eps,
        )

    def _forward(self, x, attention_mask=None):
        attention_output = self.mha(
            query=x,
            value=x,
            attention_mask=attention_mask,
        )
        attention_output = self.dropout1(attention_output)
        x = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(x)
        feedforward_output = self.dropout2(feedforward_output)
        x = self.norm2(x + feedforward_output)

        return x

    def xl_forward(
        self,
        x,
        positional_embedding,
        u_weight,
        v_weight,
        indices,
        lower,
        attention_mask=None,
    ):
        attention_output = self.mha(
            query=x,
            value=x,
            positional_embedding=positional_embedding,
            u_weight=u_weight,
            v_weight=v_weight,
            indices=indices,
            lower=lower,
            attention_mask=attention_mask,
        )
        attention_output = self.dropout1(attention_output)
        x = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(x)
        feedforward_output = self.dropout2(feedforward_output)
        x = self.norm2(x + feedforward_output)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_attention_heads,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_feedforward=nn.functional.relu,
        dim_feedforward=None,
        dropout_feedforward=0.0,
        key_dim=None,
        bias=True,
        layer_dropout=False,
        mha_type="torch",
        rotary_base=10_000,
        rope_percentage=0.5,
        norm_eps=1e-6,
        use_rms_norm=False,
    ):
        super().__init__()
        if not num_layers > 0:
            raise ValueError(f"'num_layers={num_layers}' must be greater than 0.")

        self.layer_dropout = layer_dropout
        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    activation_feedforward=activation_feedforward,
                    dim_feedforward=dim_feedforward,
                    dropout_feedforward=dropout_feedforward,
                    key_dim=key_dim,
                    bias=bias,
                    mha_type=mha_type,
                    rotary_base=rotary_base,
                    rope_percentage=rope_percentage,
                    norm_eps=norm_eps,
                    use_rms_norm=use_rms_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.forward = self._forward
        if mha_type == "xl":
            self.u_weight = nn.Parameter(
                torch.empty(
                    1,
                    1,
                    num_attention_heads,
                    hidden_size // num_attention_heads,
                )
            )
            nn.init.xavier_uniform_(self.u_weight)

            self.v_weight = nn.Parameter(
                torch.empty(
                    1,
                    1,
                    num_attention_heads,
                    hidden_size // num_attention_heads,
                )
            )
            nn.init.xavier_uniform_(self.v_weight)

            self.positional_embedding = SinusoidalEmbedding(hidden_size=hidden_size)
            self.forward = self.xl_forward

    def _forward(self, x, attention_mask=None):
        hidden_state = x
        for enc_layer in self.enc_layers:
            skip_the_layer = (
                True
                if self.training
                and self.layer_dropout
                and (torch.rand([]) < self.layer_dropout)
                else False
            )
            hidden_state = (
                hidden_state
                if skip_the_layer
                else enc_layer(hidden_state, attention_mask=attention_mask)
            )

        return hidden_state

    def xl_forward(self, x, attention_mask=None):
        indices = create_indices(x.size(1), device=x.device)
        lower = create_negative_lower(x.size(1), dtype=x.dtype, device=x.device)
        positional_embedding = self.positional_embedding(x)  ### (S, input_dim)

        hidden_state = x
        for enc_layer in self.enc_layers:
            skip_the_layer = (
                True
                if self.training
                and self.layer_dropout
                and (torch.rand([]) < self.layer_dropout)
                else False
            )
            if not skip_the_layer:
                hidden_state = enc_layer(
                    hidden_state,
                    positional_embedding=positional_embedding,
                    u_weight=self.u_weight,
                    v_weight=self.v_weight,
                    attention_mask=attention_mask,
                    indices=indices,
                    lower=lower,
                )

        return hidden_state


class CrossAtentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_feedforward=nn.functional.relu,
        dim_feedforward=None,
        dropout_feedforward=0.0,
        key_dim=None,
        value_dim=None,
        bias=True,
        mha_type="torch",
        rotary_base=10_000,
        rope_percentage=0.5,
        norm_eps=1e-6,
        use_rms_norm=False,
    ):
        super().__init__()
        self.config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
            "activation_feedforward": activation_feedforward,
            "dim_feedforward": dim_feedforward,
            "dropout_feedforward": dropout_feedforward,
            "key_dim": key_dim,
            "bias": bias,
            "mha_type": mha_type,
            "rotary_base": rotary_base,
            "rope_percentage": rope_percentage,
            "norm_eps": norm_eps,
            "use_rms_norm": use_rms_norm,
            "value_dim": value_dim,
        }
        key_dim = hidden_size // num_attention_heads if key_dim is None else key_dim

        if mha_type == "native":
            self.mha = NativeMultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                dropout=attention_dropout,
                value_dim=value_dim,
                bias=bias,
            )
        elif mha_type == "torch":
            self.mha = MultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                dropout=attention_dropout,
                value_dim=value_dim,
                bias=bias,
            )
        elif mha_type == "rotary":
            self.mha = RotaryPEMultiHeadAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                key_dim=key_dim,
                dropout=attention_dropout,
                value_dim=value_dim,
                bias=bias,
                base=rotary_base,
                rope_percentage=rope_percentage,
            )
        else:
            raise ValueError(f"mha_type= '{mha_type}' is not knwon.")

        self.feedforward = PositionwiseFF(
            hidden_size=hidden_size,
            dim_feedforward=dim_feedforward,
            activation_feedforward=activation_feedforward,
            dropout_feedforward=dropout_feedforward,
            bias=bias,
        )
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.dropout2 = nn.Dropout(hidden_dropout)

        NormClass = RMSNorm if use_rms_norm else nn.LayerNorm

        self.norm1 = NormClass(
            hidden_size,
            eps=norm_eps,
        )
        self.norm2 = NormClass(
            hidden_size,
            eps=norm_eps,
        )

    def forward(self, x, memory, attention_mask=None):
        attention_output = self.mha(
            query=x,
            value=memory,
            attention_mask=attention_mask,
        )
        attention_output = self.dropout1(attention_output)
        x = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(x)
        feedforward_output = self.dropout2(feedforward_output)
        x = self.norm2(x + feedforward_output)

        return x
