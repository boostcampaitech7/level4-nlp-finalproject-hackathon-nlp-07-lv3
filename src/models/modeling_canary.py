import os
from typing import Any, Iterable, List, Optional, Union
from dataclasses import dataclass, fields, is_dataclass

import torch
from tqdm import  tqdm
from omegaconf import OmegaConf, DictConfig
from nemo.collections.asr.models import EncDecMultiTaskModel

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

@dataclass
class InternalTranscribeConfig:
    # Internal values
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: Optional[torch.dtype] = None
    training_mode: bool = False
    logging_level: Optional[Any] = None

    # Preprocessor values
    dither_value: float = 0.0
    pad_to_value: int = 0

    # Scratch space
    temp_dir: Optional[str] = None
    manifest_filepath: Optional[str] = None

ChannelSelectorType = Union[int, Iterable[int], str]

@dataclass
class TranscribeConfig:
    batch_size: int = 4
    return_hypotheses: bool = False
    num_workers: Optional[int] = None
    channel_selector: ChannelSelectorType = None
    augmentor: Optional[DictConfig] = None
    timestamps: Optional[bool] = None  # returns timestamps for each word and segments if model supports punctuations
    verbose: bool = True

    # Utility
    partial_hypothesis: Optional[List[Any]] = None

    _internal: Optional[InternalTranscribeConfig] = None

def move_data_to_device(inputs: Any, device: Union[str, torch.device], non_blocking: bool = True) -> Any:
    """Recursively moves inputs to the specified device"""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=non_blocking)
    elif isinstance(inputs, (list, tuple, set)):
        return inputs.__class__([move_data_to_device(i, device, non_blocking) for i in inputs])
    elif isinstance(inputs, dict):
        return {k: move_data_to_device(v, device, non_blocking) for k, v in inputs.items()}
    elif is_dataclass(inputs):
        return type(inputs)(
            **{
                field.name: move_data_to_device(getattr(inputs, field.name), device, non_blocking)
                for field in fields(inputs)
            }
        )
    else:
        return inputs

def get_dataloader_from_config(model : EncDecMultiTaskModel, manifet_path : str, test_config : OmegaConf = None, transcribe_cfg: TranscribeConfig = None):
    if test_config:
        config = config
    
    else:
        config = OmegaConf.create(
        dict(
            manifest_filepath=manifet_path,
            sample_rate=16000,
            labels=None,
            batch_size=2,
            shuffle=False,
            time_length=20,
            use_lhotse = True,
        )
    )
    
    return config, model._setup_dataloader_from_config(config=config)

def get_transcribe_config(transcribe_cfg: TranscribeConfig = None):
    if transcribe_cfg:
        transcribe_cfg = transcribe_cfg
    
    else:
        transcribe_cfg = TranscribeConfig(
            batch_size=2,
            return_hypotheses=False,
            num_workers=0,
            channel_selector=None,
            augmentor=None,
            verbose=True,
            timestamps=None,
            _internal=InternalTranscribeConfig(),
        ) 

    return transcribe_cfg


def get_embeddings(speaker_model : EncDecMultiTaskModel, manifest_file, batch_size=2, embedding_dir='./', device='cuda'):
    config, _ = get_dataloader_from_config(speaker_model, manifest_file)
    transcribe_cfg = get_transcribe_config()

    speaker_model.setup_test_data(config)
    speaker_model = speaker_model.to(device)
    speaker_model.eval()
           
    for test_batch in tqdm(speaker_model.test_dataloader(), desc="Transcribing", disable=not transcribe_cfg.verbose):      
        # Move batch to device
        # Run forward pass
        with autocast():
            test_batch = move_data_to_device(test_batch, transcribe_cfg._internal.device)
            spectrogram, spectrogram_len, = test_batch.audio, test_batch.audio_lens
            spectrogram = spectrogram.to(device)
            
            log_probs, encoded_len, speech_embeds, enc_mask = speaker_model.forward(input_signal=spectrogram, input_signal_length=spectrogram_len)

    return speech_embeds

# manifest_filepath = os.path.join(os.getcwd(), 'src', 'models', 'manifest.json')
# device = 'cuda'
# # load model
# canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

# get_embeddings(canary_model, manifest_filepath, batch_size=2, embedding_dir='./', device=device)

# update dcode params
# decode_cfg = canary_model.cfg.decoding
# decode_cfg.beam.beam_size = 1
# canary_model.change_decoding_strategy(decode_cfg)

# preprocessor = canary_model.preprocessor
# if isinstance(preprocessor, AudioToMelSpectrogramPreprocessor):
#     sr = preprocessor._sample_rate
#     # result = preprocessor.process([test_path])
#     # processed_signal, processed_signal_length = self.preprocessor(
#     #     input_signal=input_signal, length=input_signal_length
#     # )

# conformer = canary_model.encoder
# n_layernorm = conformer.d_model

# for name, param in conformer.named_parameters():
#     print(name, param, param.requires_grad)

# test_path = os.path.join(os.getcwd(), 'src', 'models', '103-1240-0008.flac')

# if isinstance(canary_model, EncDecMultiTaskModel):
#     predicted_text = canary_model.transcribe(audio=[test_path])
#     print(predicted_text)