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

import torch
import torch.nn as nn
import random


class ExtendedCanaryEncoder(nn.Module):
    def __init__(self, encoder, transformer_layers, embed_positions, dropout, layer_norm):
        super().__init__()
        self.encoder = encoder  # The original NVIDIA Canary Encoder
        self.transformer_layers = transformer_layers  # List of Transformer layers
        self.embed_positions = embed_positions  # Positional embeddings
        self.dropout = dropout  # Dropout probability
        self.layer_norm = layer_norm  # Final LayerNorm
        self.training = True  # To simulate training mode, set externally in practice

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        """
        Forward pass with extended hidden state vector computation.
        """
        # Step 1: Call the original NVIDIA Canary Encoder
        transf_log_probs, encoded_len, enc_states, enc_mask = self.encoder(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            transcript=transcript,
            transcript_length=transcript_length,
        )

        # Step 2: Add positional embeddings to encoded states
        hidden_states = enc_states + self.embed_positions.weight

        # Step 3: Apply dropout to hidden states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Step 4: Initialize containers for optional outputs
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Step 5: Process through Transformer layers
        for idx, transformer_layer in enumerate(self.transformer_layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # Optional: Skip layers during training based on LayerDrop
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < 0.1):  # Adjust layerdrop probability as needed
                continue

            # Compute forward pass through the transformer layer
            layer_outputs = transformer_layer(
                hidden_states,
                attention_mask=enc_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Step 6: Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)

        # Append final hidden states to `encoder_states` if requested
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # Step 7: Return results
        return {
            "transf_log_probs": transf_log_probs,
            "encoded_len": encoded_len,
            "last_hidden_state": hidden_states,
            "hidden_states": encoder_states,
            "attentions": all_attentions,
            "encoder_mask": enc_mask,
        }


@dataclass
class InternalTranscribeConfig:
    # Internal values
    device: Optional[torch.device] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    channel_selector: ChannelSelectorType = 0
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

def get_dataloader_from_config(model : EncDecMultiTaskModel, manifet_path : str, batch_size, test_config : OmegaConf = None):
    if test_config:
        config = config

    else:
        config = OmegaConf.create(
        dict(
            manifest_filepath=manifet_path,
            sample_rate=16000,
            labels=None,
            batch_size=batch_size,
            shuffle=False,
            time_length=30,
            use_lhotse = True,
            channel_selector=0,
        )
    )

    return config, model._setup_dataloader_from_config(config=config)

def get_transcribe_config(manifest_filepath, batch_size, transcribe_cfg: TranscribeConfig = None):
    if transcribe_cfg:
        transcribe_cfg = transcribe_cfg

    else:
        transcribe_cfg = TranscribeConfig(
            batch_size=batch_size,
            return_hypotheses=False,
            num_workers=0,
            channel_selector=0,
            augmentor=None,
            verbose=True,
            timestamps=None,
            _internal=InternalTranscribeConfig(
                device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                dtype = None,
                training_mode = False,
                logging_level = None,

                # Preprocessor values
                dither_value = 0.0,
                pad_to_value = 0,

                # Scratch space
                temp_dir = None,
                manifest_filepath = manifest_filepath,
            ),
        )

    return transcribe_cfg

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
