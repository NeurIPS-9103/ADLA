
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.layers import trunc_normal_


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):


        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings




def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class DiffAttention(nn.Module):
    """Diff Attention adapted for DiT patch tokens."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., block_depth=None, input_resolution=None, **kwargs):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for DiffAttention")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.lambda_init = lambda_init_fn(block_depth)
        half_head_dim = self.head_dim // 2
        self.lambda_q1 = nn.Parameter(torch.zeros(half_head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(half_head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(half_head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(half_head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.block_num_scale = 3
        init_spatial_side = input_resolution[0] if input_resolution[0] == input_resolution[1] \
            else int(round(math.sqrt(input_resolution[0] * input_resolution[1])))
        _, self.block_pos_tokens = self._compute_num_blocks(init_spatial_side)
        self.block_pos_dim = self.head_dim // 2
        self.block_q_embed = nn.Parameter(
            torch.randn(self.num_heads, self.block_pos_tokens, self.block_pos_dim)
        )
        self.block_k_embed = nn.Parameter(
            torch.randn(self.num_heads, self.block_pos_tokens, self.block_pos_dim)
        )

        self.nf = 8
        self.alpha = 2
        self.router = nn.Linear(dim, self.nf)
        self.power = nn.Parameter(torch.zeros(self.nf))
        self.eps = 1e-6

        self.diff_norm = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)

    def _compute_num_blocks(self, spatial_side):
        base_blocks_per_side = max(1, int(math.sqrt(spatial_side)))
        base_num_blocks = base_blocks_per_side * base_blocks_per_side
        target_num_blocks = max(1, int(round(base_num_blocks * self.block_num_scale)))
        blocks_per_side = max(1, int(round(math.sqrt(target_num_blocks))))
        return blocks_per_side, blocks_per_side * blocks_per_side

    def _resize_block_pos(self, pos, target_tokens):
        if pos.shape[1] == target_tokens:
            return pos
        return F.interpolate(
            pos.permute(0, 2, 1),
            size=target_tokens,
            mode='linear',
            align_corners=False,
        ).permute(0, 2, 1)

    @staticmethod
    def _infer_special_tokens(n_tokens):
        for num_special in (1, 0):
            spatial_tokens = n_tokens - num_special
            if spatial_tokens <= 0:
                continue
            spatial_side = int(math.isqrt(spatial_tokens))
            if spatial_side * spatial_side == spatial_tokens:
                return num_special, spatial_side
        return 0, -1

    def _global_linear_attn(self, q, k, v):
        n = q.shape[-2]
        z = 1.0 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + self.eps)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        return q @ kv * z

    def _block_weighted_linear_attn(self, q, k, v):
        b, h, n, _ = q.shape
        _, spatial_side = self._infer_special_tokens(n)
        if spatial_side <= 0:
            return self._global_linear_attn(q, k, v)

        num_special = n - spatial_side * spatial_side
        n_spatial = spatial_side * spatial_side
        if n_spatial <= 0:
            return self._global_linear_attn(q, k, v)

        q_special = q[:, :, :num_special, :] if num_special > 0 else None
        q_spatial = q[:, :, num_special:, :]
        k_spatial = k[:, :, num_special:, :]
        v_spatial = v[:, :, num_special:, :]

        blocks_per_side, num_blocks = self._compute_num_blocks(spatial_side)

        pad_h = (blocks_per_side - (spatial_side % blocks_per_side)) % blocks_per_side
        side_pad = spatial_side + pad_h
        block_h = side_pad // blocks_per_side
        tokens_per_block = block_h * block_h

        q_spatial = q_spatial.reshape(b, h, spatial_side, spatial_side, -1)
        k_spatial = k_spatial.reshape(b, h, spatial_side, spatial_side, -1)
        v_spatial = v_spatial.reshape(b, h, spatial_side, spatial_side, -1)

        if pad_h > 0:
            q_spatial = F.pad(q_spatial, (0, 0, 0, pad_h, 0, pad_h))
            k_spatial = F.pad(k_spatial, (0, 0, 0, pad_h, 0, pad_h))
            v_spatial = F.pad(v_spatial, (0, 0, 0, pad_h, 0, pad_h))

        def to_block_tokens(t):
            t = t.reshape(b, h, blocks_per_side, block_h, blocks_per_side, block_h, -1)
            t = t.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, h, num_blocks, tokens_per_block, -1)
            return t

        q_block_tokens = to_block_tokens(q_spatial)
        k_block_tokens = to_block_tokens(k_spatial)
        v_block_tokens = to_block_tokens(v_spatial)

        q_block = q_block_tokens.mean(dim=-2)
        k_block = k_block_tokens.mean(dim=-2)
        q_pos = self._resize_block_pos(self.block_q_embed, num_blocks).to(device=q_block.device, dtype=q_block.dtype)
        k_pos = self._resize_block_pos(self.block_k_embed, num_blocks).to(device=k_block.device, dtype=k_block.dtype)
        q_block = q_block + q_pos.unsqueeze(0)
        k_block = k_block + k_pos.unsqueeze(0)

        block_logits = (q_block @ k_block.transpose(-2, -1)) * (q_block.shape[-1] ** -0.5)
        block_attn = self.attn_drop(torch.softmax(block_logits, dim=-1))

        scale = tokens_per_block ** -0.5
        kv_summary = torch.einsum(
            'bhmtd,bhmte->bhmde',
            k_block_tokens * scale,
            v_block_tokens * scale,
        )
        k_summary = k_block_tokens.mean(dim=-2)

        kv_weighted = torch.einsum('bhmn,bhnde->bhmde', block_attn, kv_summary)
        k_weighted = torch.einsum('bhmn,bhnd->bhmd', block_attn, k_summary)

        x_spatial = torch.einsum('bhmtd,bhmde->bhmte', q_block_tokens, kv_weighted)
        z_spatial = torch.einsum('bhmtd,bhmd->bhmt', q_block_tokens, k_weighted).unsqueeze(-1)
        x_spatial = x_spatial / (z_spatial + self.eps)

        x_spatial = x_spatial.reshape(
            b, h, blocks_per_side, blocks_per_side, block_h, block_h, -1
        ).permute(0, 1, 2, 4, 3, 5, 6).reshape(b, h, side_pad, side_pad, -1)
        x_spatial = x_spatial[:, :, :spatial_side, :spatial_side, :].reshape(b, h, n_spatial, -1)

        if num_special <= 0:
            return x_spatial

        special_logits = (q_special @ k_block.transpose(-2, -1)) * (q_block.shape[-1] ** -0.5)
        special_attn = self.attn_drop(torch.softmax(special_logits, dim=-1))
        kv_special = torch.einsum('bhsm,bhmde->bhsde', special_attn, kv_summary)
        k_special = torch.einsum('bhsm,bhmd->bhsd', special_attn, k_summary)
        x_special = torch.einsum('bhsd,bhsde->bhse', q_special, kv_special)
        z_special = torch.einsum('bhsd,bhsd->bhs', q_special, k_special).unsqueeze(-1)
        x_special = x_special / (z_special + self.eps)

        return torch.cat([x_special, x_spatial], dim=2)

    def forward(self, x):
        b, n, c = x.shape

        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        route_logits = self.router(x)
        if self.training:
            route_weights = F.gumbel_softmax(route_logits, tau=1.0, hard=True, dim=-1)
        else:
            route_index = route_logits.argmax(dim=-1, keepdim=True)
            route_weights = torch.zeros_like(route_logits).scatter_(-1, route_index, 1.0)

        actual_powers = 1.0 + self.alpha * torch.sigmoid(self.power)
        token_powers = (route_weights @ actual_powers.unsqueeze(-1)).view(b, 1, n, 1)

        q = torch.relu(q) + self.eps
        k = torch.relu(k) + self.eps

        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** token_powers
        k = k ** token_powers
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        x1 = self._block_weighted_linear_attn(q1, k1, v)
        x2 = self._global_linear_attn(q2, k2, v)

        lambda_term1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_term2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_term1 - lambda_term2 + self.lambda_init

        x = x1 - x2 * lambda_full
        x = self.diff_norm(x) * (1 - self.lambda_init)
        x = x.transpose(1, 2).reshape(b, n, c)

        h = w = int(math.sqrt(n))
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        block_depth=None,
        attn_type="diff",
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == "diff":
            self.attn = DiffAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=0.,
                proj_drop=0.,
                block_depth=block_depth,
                input_resolution=window_size,
                **block_kwargs,
            )
        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        attn_type="diff",
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_type = attn_type

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads,
                window_size=(input_size//patch_size, input_size//patch_size),
                mlp_ratio=mlp_ratio,
                block_depth=i,
                attn_type=attn_type,
            ) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
