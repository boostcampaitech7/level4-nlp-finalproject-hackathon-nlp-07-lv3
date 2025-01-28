import torch
from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
from einops.layers.torch import Rearrange

from src.models.CED.models.audiotransformer import Attention, Block
from .CED.models.audiotransformer import *

class ExtendedCEDEncoder(AudioTransformer):

    def __init__(self,
                 outputdim=527,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_bn: bool = True,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1012,
                 pooling='mean',
                 wavtransforms=None,
                 spectransforms=None,
                 time_patch_out: Optional[float] = None,
                 freq_patch_out: Optional[float] = None,
                 block_type=Block,
                 attention_type=Attention,
                 eval_avg='mean',
                 **kwargs):
        super().__init__(outputdim, patch_size, patch_stride, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, init_bn, norm_layer, act_layer, init_values, target_length, pooling, wavtransforms, spectransforms, time_patch_out, freq_patch_out, block_type, attention_type, eval_avg, **kwargs)

    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.init_bn(x)
        if x.shape[-1] > self.maximal_allowed_length:
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(*x.shape[:-1],
                                      self.target_length,
                                      device=x.device)
                    pad[..., :splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = rearrange(splits, 'spl b c f t-> (spl b) c f t')
            x = self.forward_head(self.forward_features(x))
            x = rearrange(x, '(spl b) d -> spl b d', spl=n_splits)
            if self.eval_avg == 'mean':
                x = x.mean(0)
            elif self.eval_avg == 'max':
                x = x.max(0)[0]
            else:
                raise ValueError(
                    f'Unknown Eval average function ({self.eval_avg})')

        else:
            x = self.forward_features(x)
        return x

    def forward(self, x):
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        if self.training:
            x = self.spectransforms(x)
        x = self.forward_spectrogram(x)
        return x