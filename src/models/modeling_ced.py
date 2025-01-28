import torch
from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
from einops.layers.torch import Rearrange
import sys

sys.path.append('/data/pgt/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/models/CED/models')
# from .CED.models.audiotransformer import Attention, Block
from .CED.models import *

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
        print("2. {}".format(x.shape))
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
            x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        return x

    def forward(self, x):
        print("1. {}".format(x.shape))
        x = x.float()
        # if self.training:
        #     x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        # if self.training:
        #     x = self.spectransforms(x)
        x = self.forward_spectrogram(x)
        return x
    


@register_model
def audiotransformer_tiny(num_classes: int = 527,
                          pretrained=False,
                          pretrained_url: str = 'https://zenodo.org/records/8275347/files/audiotransformer_tiny_mae_as_10s.pt?download=1',
                          **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def ced_tiny(
        num_classes: int = 527,
        pretrained=True,
        pretrained_url:
    str = 'https://zenodo.org/record/8275319/files/audiotransformer_tiny_mAP_4814.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def audiotransformer_mini(
        num_classes: int = 527,
        pretrained=True,
        pretrained_url:
    str = 'https://zenodo.org/record/8275347/files/audiotransformer_mini_mae_as_10s.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=256,
                        depth=12,
                        num_heads=4,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def ced_mini(
        num_classes: int = 527,
        pretrained=True,
        pretrained_url:
    str = 'https://zenodo.org/record/8275319/files/audiotransformer_mini_mAP_4896.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=256,
                        depth=12,
                        num_heads=4,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def audiotransformer_small(
        num_classes: int = 527,
        pretrained=True,
        pretrained_url:
    str = 'https://zenodo.org/record/8275347/files/audiotransformer_small_mae_as_10s.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=6,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def ced_small(
        num_classes: int = 527,
        pretrained=True,
        pretrained_url:
    str = 'https://zenodo.org/record/8275319/files/audiotransformer_small_mAP_4958.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=384,
                        depth=12,
                        num_heads=6,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def audiotransformer_base(
        num_classes: int = 527,
        pretrained=True,
        pretrained_url:
    str = 'https://zenodo.org/record/8275347/files/audiotransformer_base_mae_as_10s.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def ced_base(
        num_classes: int = 527,
        pretrained=False,
        pretrained_url:
    str = 'https://zenodo.org/record/8275319/files/audiotransformer_base_mAP_4999.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def audiotransformer_base_4740(
        num_classes: int = 527,
        pretrained=True,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/audiotransformer_base_mAP_47_40.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4,
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        ExtendedCEDEncoder,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


if __name__ == "__main__":
    ced_mini()
