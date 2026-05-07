

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

from .deit import Block

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth) if depth is not None else 0.8

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            norm_x = norm_x * self.weight
        return norm_x

class DiffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 block_depth=None, lambda_warmup_steps=25000, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Block settings: target number of blocks is about 3x original block count.
        self.block_num_scale = 3.0
        # You can edit this single value if you want a different base block-token count.
        self.block_pos_tokens = 25 
        self.block_pos_dim = self.head_dim
        self.block_q_embed = nn.Parameter(
            torch.randn(self.num_heads, self.block_pos_tokens, self.block_pos_dim)
        )
        self.block_k_embed = nn.Parameter(
            torch.randn(self.num_heads, self.block_pos_tokens, self.block_pos_dim)
        )

        self.nf = 8
        self.alpha = 4
        self.router = nn.Linear(dim, self.nf)
        self.power = nn.Parameter(torch.zeros(self.nf))

        self.eps = 1e-6

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                        padding=1, groups=dim)

    def _lambda_warmup_factor(self):
        if self.lambda_warmup_steps <= 0:
            return 1.0
        if self.training:
            self.lambda_warmup_step.add_(1)
        return torch.clamp(
            self.lambda_warmup_step.float() / float(self.lambda_warmup_steps),
            max=1.0,
        )

    def _resize_block_pos(self, pos, target_tokens):
        if pos.shape[1] == target_tokens:
            return pos
        # Interpolate on token axis to match dynamic block count.
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
        """
        q, k: [B, H, N, Dq]
        v:    [B, H, N, Dv]
        """
        b, h, n, _ = q.shape
        _, spatial_side = self._infer_special_tokens(n)
        if spatial_side <= 0:
            return 0

        num_special = n - spatial_side * spatial_side
        n_spatial = spatial_side * spatial_side
        if n_spatial <= 0:
            return 0

        q_special = q[:, :, :num_special, :] if num_special > 0 else None
        q_spatial = q[:, :, num_special:, :]
        k_spatial = k[:, :, num_special:, :]
        v_spatial = v[:, :, num_special:, :]

        base_blocks_per_side = max(1, int(math.sqrt(spatial_side)))
        base_num_blocks = base_blocks_per_side * base_blocks_per_side
        target_num_blocks = max(1, int(round(base_num_blocks * self.block_num_scale)))
        blocks_per_side = max(1, int(round(math.sqrt(target_num_blocks))))
        num_blocks = blocks_per_side * blocks_per_side  

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
        q_pos = self._resize_block_pos(self.block_q_embed, num_blocks).to(
            device=q_block.device, dtype=q_block.dtype
        )
        k_pos = self._resize_block_pos(self.block_k_embed, num_blocks).to(
            device=k_block.device, dtype=k_block.dtype
        )
        q_block = q_block + q_pos.unsqueeze(0)
        k_block = k_block + k_pos.unsqueeze(0)
        block_logits = (q_block @ k_block.transpose(-2, -1)) * (q_block.shape[-1] ** -0.5)
        block_attn = self.attn_drop(F.softmax(block_logits, dim=-1))

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
        special_attn = self.attn_drop(F.softmax(special_logits, dim=-1))
        kv_special = torch.einsum('bhsm,bhmde->bhsde', special_attn, kv_summary)
        k_special = torch.einsum('bhsm,bhmd->bhsd', special_attn, k_summary)
        x_special = torch.einsum('bhsd,bhsde->bhse', q_special, kv_special)
        z_special = torch.einsum('bhsd,bhsd->bhs', q_special, k_special).unsqueeze(-1)
        x_special = x_special / (z_special + self.eps)

        return torch.cat([x_special, x_spatial], dim=2)

        
    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v: b, n, c


        route_logits = self.router(x)
        if self.training:
            route_weights = F.gumbel_softmax(route_logits, tau=1.0, hard=True, dim=-1)
        else:
            route_index = route_logits.argmax(dim=-1, keepdim=True)
            route_weights = torch.zeros_like(route_logits).scatter_(-1, route_index, 1.0)
        actual_powers = 1.0 + self.alpha * torch.sigmoid(self.power)
        token_powers = (route_weights @ actual_powers.unsqueeze(-1)).view(b, 1, n, 1)

        kernel_function = nn.ReLU()
        q = kernel_function(q) + self.eps
        k = kernel_function(k) + self.eps

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # b, num_heads, n, head_dim
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # b, num_heads, n, head_dim
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # b, num_heads, n, head_dim

        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** token_powers
        k = k ** token_powers
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm


        x = self._block_weighted_linear_attn(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        h = w = int(n ** 0.5)
        v_ = v[:, :, 1:, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x[:, 1:, :] = x[:, 1:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n - 1, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiffBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 block_depth=None, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DiffAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=drop, block_depth=block_depth, **kwargs
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 diff_layer=-1, lambda_warmup_steps=40000):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            diff_layer: (int): number of layers to use diff attention (-1 means all layers)
            lambda_warmup_steps: (int): training steps for lambda warmup in DiffAttention
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        diff_layer = diff_layer if diff_layer > 0 else depth

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            DiffBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                block_depth=i, lambda_warmup_steps=lambda_warmup_steps
            ) if i < diff_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):

    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = _cfg(url='', num_classes=kwargs.get('num_classes', 1000))
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if pretrained and not default_cfg.get('url'):
        _logger.warning(
            f"pretrained=True for {variant}, but no default_cfg URL is available; "
            "skipping pretrained weight download."
        )

    default_num_classes = default_cfg.get('num_classes', 1000)
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg.get('url', ''),
        **kwargs)
    return model



@register_model
def adla_deit_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def adla_deit_small(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def adla_deit_base(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
