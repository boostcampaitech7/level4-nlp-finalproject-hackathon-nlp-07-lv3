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

import json
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor


class SALMONNDataset(Dataset):
    def __init__(self, prefix, ann_path, whisper_path):
        super().__init__()

        # 경로 검증
        ann_path = prefix + ann_path
        if not os.path.exists(ann_path):
            raise ValueError(
                f"Provided prefix path '{prefix}' is not an absolute path. "
                "Please provide an absolute path to the dataset. fix run.prefix which is in configs/train.yaml"
            )

        self.prefix = prefix
        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.annotation)

    def _get_audio_path(self, path):
        # 경로를 정규화하여 // 또는 / 문제를 해결
        return os.path.normpath(os.path.join(self.prefix, path.lstrip('/').lstrip('\\')))

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_path = self._get_audio_path(ann["path"])

        try:
            audio, sr = sf.read(audio_path)
        except sf.LibsndfileError as e:
            logging.error(f"LibsndfileError: {e}")
            logging.error(f"Exception details: {e.args}")
            logging.error(f"Failed to load audio file: {audio_path}")

            # 0번째 샘플을 로드 시도
            audio, sr = self._load_fallback_audio()
            if audio.size == 0:
                logging.warning("Returning empty audio due to failure in loading fallback audio.")

        if len(audio.shape) == 2:  # stereo to mono
            audio = audio[:, 0]

        if len(audio) < sr:  # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)

        if sr != self.wav_processor.sampling_rate:  # TODO. use more efficient implementation
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.wav_processor.sampling_rate)
            sr = self.wav_processor.sampling_rate

        audio = audio[: sr * 30]  # truncate audio to at most 30s

        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }

    def _get_audio_path(self, path):
        # 경로를 정규화하여 // 또는 / 문제를 해결
        return os.path.normpath(os.path.join(self.prefix, path.lstrip('/').lstrip('\\')))

    def _load_fallback_audio(self):
        try:
            fallback_path = self._get_audio_path(self.annotation[0]["path"])
            print(f"Failed to load original audio. Loading 0-th sample instead: {fallback_path}")
            return sf.read(fallback_path)
        except (IOError, sf.SoundFileError) as e:
            print(f"Failed to load 0-th sample as well: {e}. Returning empty audio.")
            return np.array([]), 44100  # 빈 오디오와 기본 샘플레이트 반환
