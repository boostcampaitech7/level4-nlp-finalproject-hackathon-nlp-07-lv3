import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

# Custom modules
from models.salmonn import SALMONN


def load_preprocessor(cfg):
    salmonn_preprocessor = SALMONN.from_config(cfg.config.model)
    salmonn_preprocessor.to(cfg.config.run.device)
    salmonn_preprocessor.eval()
    return salmonn_preprocessor


def load_model(salmonn_preprocessor):
    model = salmonn_preprocessor.qwen_model
    tokenizer = salmonn_preprocessor.qwen_tokenizer
    return model, tokenizer


class SALMONNTestDataset(Dataset):
    def __init__(self, prefix, ann_path, whisper_path, task=None):
        super().__init__()

        self.prefix = prefix

        ann_path = prefix + ann_path
        if not os.path.exists(ann_path):
            raise ValueError(
                f"Provided prefix path '{prefix}' is not an absolute path. "
                "Please provide an absolute path to the dataset. fix run.prefix which is in configs/train.yaml"
            )

        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.task = task

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

        testset_id = [s["testset_id"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        entity = {
            "testset_id": testset_id,
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "task": task,
            "Q": Q,
            "id": id,
        }

        # evaluate의 경우에는 text가 없으므로 없는 경우 pass 그외에는 전부 있음
        if self.task is not None:
            try:
                entity["text"] = [s["text"] for s in samples]

            except KeyError:
                pass

        return entity

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_path = self._get_audio_path(ann["path"])

        try:
            audio, sr = sf.read(audio_path)
        except (RuntimeError, IOError) as e:
            print(f"Failed to load {audio_path}. Error: {e}. Loading 0-th sample instead.")
            audio, sr = sf.read(os.path.join(self.prefix, self.annotation[0]["path"]))

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
        testset_id = ann["testset_id"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        entity = {
            "testset_id": testset_id,
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }

        # evaluate의 경우에는 text가 없으므로 없는 경우 pass 그외에는 전부 있음
        if self.task is not None:
            try:
                entity["text"] = ann["text"]

            except KeyError:
                pass

        return entity
