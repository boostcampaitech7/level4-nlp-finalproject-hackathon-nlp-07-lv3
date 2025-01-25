# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers.modeling_outputs import CausalLMOutputWithPast

from dist_utils import get_rank, get_world_size, is_main_process


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_dataloader(dataset, config, is_train=True, use_distributed=True):
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train, num_replicas=get_world_size(), rank=get_rank())
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None and is_train,
        collate_fn=dataset.collater,
        drop_last=is_train, # true이면 batch 사이즈보다 작은 나머지들은 그냥 버림
    )

    if is_train:
        loader = IterLoader(loader, use_distributed=use_distributed)

    return loader


def apply_to_sample(f, sample, device="cuda:0"):
    if len(sample) == 0:
        return {}

    def _apply(x, device):
        if torch.is_tensor(x):
            return f(x, device)
        elif isinstance(x, dict):
            return {key: _apply(value, device) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x, device) for x in x]
        elif isinstance(x, CausalLMOutputWithPast):
            return CausalLMOutputWithPast(
                logits=x.logits.to(device) if x.logits is not None else None,
                past_key_values=tuple(
                    tuple(p.to(device) for p in pkv) for pkv in x.past_key_values
                ) if x.past_key_values is not None else None,
                hidden_states=x.hidden_states.to(device) if x.hidden_states is not None else None,
                attentions=x.attentions.to(device) if x.attentions is not None else None,
                loss=x.loss.to(device) if x.loss is not None else None,
            )
        else:
            return x

    return _apply(sample, device)


def move_to_cuda(sample, device="cuda:0"):
    def _move_to_cuda(tensor, device):
        return tensor.to(device)

    return apply_to_sample(_move_to_cuda, sample, device)


def prepare_sample(samples, cuda_enabled=True, device="cuda:0"):
    if cuda_enabled:
        samples = move_to_cuda(samples, device)

    # TODO fp16 support

    return samples


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def prepare_one_sample(wav_path, wav_processor, cuda_enabled=True):
    audio, sr = sf.read(wav_path)
    if len(audio.shape) == 2:  # stereo to mono
        audio = audio[:, 0]
    if len(audio) < sr:  # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)

    if sr != wav_processor.sampling_rate:  # TODO. use more efficient implementation
        audio = librosa.resample(audio, orig_sr=sr, target_sr=wav_processor.sampling_rate)
        sr = wav_processor.sampling_rate

    audio = audio[: sr * 30]  # truncate audio to at most 30s

    spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]

    samples = {
        "spectrogram": spectrogram,
        "raw_wav": torch.from_numpy(audio).unsqueeze(0),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
    }
    if cuda_enabled:
        samples = move_to_cuda(samples)

    return samples
