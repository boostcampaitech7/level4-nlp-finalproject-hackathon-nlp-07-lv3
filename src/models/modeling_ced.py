import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast
import torchaudio.transforms as audio_transforms

from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
from einops.layers.torch import Rearrange

from .CED.models import *

class FrontEnd(nn.Sequential):

    def __init__(self,
                 f_min: int = 0,
                 sample_rate: int = 16000,
                 win_size: int = 512, # frame_length, BEATs에선 25ms 즉, 400 (본래 CED에서는 512)
                 center: bool = True,
                 n_fft: int = 512, # 주파수 분해능 (주파수 해상도 결정, n_fft ≥ win_length 절대 만족 FFT 시 윈도우 데이터 손실 방지), BEATs에선 25ms 즉, 400 (본래 CED에서는 512)
                 f_max: Optional[int] = None,
                 hop_size: int = 160, # 프레임 간격 (프레임 간 겹침량 결정), 예시) 프레임 겹침 = win_length - hop_length (예: 400샘플 윈도우 - 160 hop → 240샘플(15ms) 겹침)
                 n_mels: int = 64):
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size
        self.n_mels = n_mels

        super().__init__(
            audio_transforms.MelSpectrogram(f_min=self.f_min,
                                            sample_rate=self.sample_rate,
                                            win_length=self.win_size,
                                            center=self.center,
                                            n_fft=self.n_fft,
                                            f_max=self.f_max,
                                            hop_length=self.hop_size,
                                            n_mels=self.n_mels),
            audio_transforms.AmplitudeToDB(top_db=120))
        
        self.register_buffer('fbank_mean', torch.tensor(15.41663))
        self.register_buffer('fbank_std', torch.tensor(6.55582))

    # Disable Autocast for FP16 training!
    @autocast(enabled=False)
    def forward(self, x):
        x = super().forward(x)
        return (x - self.fbank_mean) / (2 * self.fbank_std) # BEATs와 동일 스케일링


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
                 target_length=3001, # Salmonn의 dataloader에서 input최대 30초로 상한 고정, (target_length 원래는 1012) 
                                     # T = floor[(L + 2*pad - win_length) / hop_length] + 1,  pad = n_fft//2 = 256
                                     # ex) 30초 오디오 → (30s * 16000) = 480,000 샘플
                                     # L + 2 * pad = 480,000 + 512 = 480,512  
                                     # 480,512 - 400 = 480,112  
                                     # 480,112 / 160 = 3,000.7 → floor(3,000.7) = 3,000  
                                     # T = 3,000 + 1 = 3,001  
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

        self.front_end = FrontEnd()


    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.init_bn(x)
        # print("2. {}".format(x.shape))
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
        # print("1. {}".format(x.shape))
        x = x.float()
        # x = pad_audio(x, self.x.shape[-1])
        x = self.front_end(x)
        x = self.forward_spectrogram(x)
        return x
    
def pad_audio(waveform, target_samples=480000):
    if waveform.size(-1) < target_samples:
        return F.pad(waveform, (0, target_samples - waveform.size(-1)))
    else:
        return waveform[:, :target_samples]

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
