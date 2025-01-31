import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import DynamicCache
from transformers import WhisperFeatureExtractor

from config import Config
from dataset import SALMONNDataset
from models.salmonn import SALMONN
from utils import get_dataloader, prepare_sample
from models.modeling_canary import get_dataloader_from_config


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(salmonn_preprocessor):
    model = salmonn_preprocessor.llama_model
    tokenizer = salmonn_preprocessor.llama_tokenizer
    return model, tokenizer


def load_preprocessor(cfg):
    salmonn_preprocessor = SALMONN.from_config(cfg.config.model)
    salmonn_preprocessor.to(cfg.config.run.device)
    salmonn_preprocessor.eval()
    return salmonn_preprocessor


class MockDataset(SALMONNDataset):
    def __init__(self, cfg, sr, audio_length, dataset_length):
        self.sr = sr
        self.audio_length = audio_length
        self.dataset_length = dataset_length
        self.prefix = cfg.config.datasets.prefix
        self.random_sample = np.random.randn(self.sr * self.audio_length)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        audio = self.random_sample.copy()
        return {
            "raw_wav": audio,
            "text": "test",
            "task": "asr",
            "Q": "",
            "id": idx,
        }

    @staticmethod
    def make_mock_dataloader(cfg, sr, audio_length, dataset_length=100):
        dataset = MockDataset(cfg, sr, audio_length, dataset_length)
        return get_dataloader(dataset, cfg.config.run, is_train=False, use_distributed=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        type=str,
        help="path to configuration file",
        default="/root/np-app-audiolm-evaluator/salmonn_eval_config.yaml",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument("--num_it", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=10)
    return parser.parse_args()


def get_gpu_memory_usage():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    gpu_memory = int(result.strip().split("\n")[0])
    return gpu_memory


def model_inference(cfg, samples, n_samples, test_prompt, salmonn):
    # TTFT
    start_time = time.time()
    llm = salmonn.llama_model

    batch_size = n_samples.audio.shape[0]
    raw_wav = samples.get("raw_wav", None)
    audio_padding_mask = samples.get("padding_mask", None)
    speech_embeds, speech_atts = salmonn.encode_speech(
        n_samples, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
    )

    prompts = [test_prompt[task] for task in samples["task"]]
    templated_prompts = [cfg.config.model.prompt_template.format(prompt) for prompt in prompts]

    speech_embeds, speech_atts = salmonn.prompt_wrap(speech_embeds, speech_atts, templated_prompts, multi_prompt=True)

    bos = (
        torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        )
        * salmonn.llama_tokenizer.bos_token_id
    )
    bos_embeds = llm.model.embed_tokens(bos) if not salmonn.lora else llm.model.model.embed_tokens(bos)
    atts_bos = speech_atts[:, :1]

    speech_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
    speech_atts = torch.cat([atts_bos, speech_atts], dim=1)

    outputs = llm.model(
        inputs_embeds=speech_embeds,
        attention_mask=speech_atts,
    )
    end_time = time.time()
    ttft = end_time - start_time

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
    past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

    # TPOT
    start_time = time.time()
    with torch.no_grad():
        _ = llm.model(next_token, past_key_values=past_key_values, use_cache=True)
    end_time = time.time()
    tpot = end_time - start_time

    inference_time = ttft + tpot
    return inference_time, ttft, tpot


def main(args):
    cfg = Config(args)

    print("Force batch size as 1")
    cfg.config.run.batch_size_eval = 1

    # Load model
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, _ = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    test_config, n_loader_test = get_dataloader_from_config(salmonn_preprocessor.speech_encoder, cfg.config.datasets.mock_manifest_path, batch_size=cfg.config.run.batch_size_eval)

    salmonn_preprocessor.speech_encoder._update_dataset_config(dataset_name='test', config=test_config)
    salmonn_preprocessor.speech_encoder._test_dl = n_loader_test

    # Load dataset
    with open(cfg.config.model.test_prompt_path, "r", encoding='utf-8') as f:
        test_prompt = json.load(f)
    dataloader = MockDataset.make_mock_dataloader(cfg, sr=16000, audio_length=10)
    sample_batch = next(iter(dataloader))
    sample_batch = prepare_sample(sample_batch, cuda_enabled=torch.cuda.is_available())

    n_samples = next(n_loader_test._get_iterator())
    n_samples = salmonn_preprocessor.move_data_to_device(n_samples, 'cuda')

    # Measure memory and latency
    memory_usages = []
    inference_times = []
    ttfts = []
    tpots = []

    for it in tqdm(range(args.num_it + args.num_warmup)):
        torch.cuda.synchronize()
        with torch.no_grad():
            inference_time, ttft, tpot = model_inference(
                cfg,
                sample_batch,
                n_samples,
                test_prompt,
                salmonn_preprocessor,
            )
        torch.cuda.synchronize()
        after_memory_allocated = torch.cuda.max_memory_allocated(device='cuda')

        torch.cuda.empty_cache()  # Clear the cache to get more accurate measurements
        gc.collect()

        if it >= args.num_warmup:
            memory_usages.append(after_memory_allocated)
            inference_times.append(inference_time)
            ttfts.append(ttft)
            tpots.append(tpot)

    average_memory_usage = np.mean(memory_usages)
    average_inference_time = np.mean(inference_times)
    average_ttft = np.mean(ttfts)
    average_tpot = np.mean(tpots)

    print(f"Average memory used during inference: {average_memory_usage/1024**3:.4f} GB")
    print(f"Average inference time: {average_inference_time:.4f} seconds")
    print(f"Average TTFT: {average_ttft:.4f} seconds")
    print(f"Average TPOT: {average_tpot:.4f} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)
