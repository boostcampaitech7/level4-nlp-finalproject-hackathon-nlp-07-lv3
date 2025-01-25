import argparse
import datetime
import os
import copy
import random
from tqdm import tqdm

import numpy as np
import pytz
import torch
import torch.backends.cudnn as cudnn
import wandb
from safetensors.torch import save_file, load_file
from torch.utils.data import random_split
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from torch.cuda.amp import autocast

from src.models import load_model
from src.config import DistillConfig

# def parse_args():
#     parser = argparse.ArgumentParser(description="train parameters")
#     parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#     parser.add_argument("--dryrun", action="store_true", help="if True, use dummy model and skip forward/backward")

#     return parser.parse_args()

# args = parse_args()
# cfg = DistillConfig(args)

# model_T_config = cfg.config.model_T
# model_S_config = cfg.config.model_S

# model_T = load_model(model_T_config).to('cuda')
# model_S = load_model(model_S_config).to('cuda')

device = 'cuda'

model1 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", # "Qwen/Qwen-1_8B",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map={"": 0},
    trust_remote_code=True,
    token="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",
)

model2 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map={"": 0},
    token="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",
)

model1_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=False, trust_remote_code=True, token="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")
model2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=False, token="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")

# model1_tokenizer.pad_token = model1_tokenizer.eos_token
# Ensure pad_token is set for model1_tokenizer
model1_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# model1_tokenizer.padding_side = "right"
model2_tokenizer.padding_side = "right"

model1.resize_token_embeddings(len(model1_tokenizer))
model2.resize_token_embeddings(len(model2_tokenizer))

text = "오늘은 점심으로 무엇이 나왔나요?"
target = "오늘은 맛있는 피자와 빵이 나왔답니다. 그리고 후식으로 초코 우유도 나왔어요."

# Concatenate input and target
combined_text = text + " " + target

# Tokenize the combined text
combined_inputs1 = model1_tokenizer(
    combined_text,
    return_tensors="pt",
    padding="longest",
    truncation=True,
    max_length=300,
    add_special_tokens=True,  # Ensure special tokens are added
).to(device)


combined_inputs2 = model2_tokenizer(
    combined_text,
    return_tensors="pt",
    padding="longest",
    truncation=True,
    max_length=300,
    add_special_tokens=True,  # Ensure special tokens are added
).to(device)


# Create labels: -100 for input tokens and target token IDs for target tokens
input_length1 = model1_tokenizer(text, return_tensors="pt").input_ids.shape[1]
labels1 = combined_inputs1['input_ids'].clone()
labels1[:, :input_length1] = -100  # Mask the input tokens


# Create labels: -100 for input tokens and target token IDs for target tokens
input_length2 = model2_tokenizer(text, return_tensors="pt").input_ids.shape[1]
labels2 = combined_inputs2['input_ids'].clone()
labels2[:, :input_length2] = -100  # Mask the input tokens

print(combined_inputs1['input_ids'].shape)
print(labels1.shape)

print(combined_inputs2['input_ids'].shape)
print(labels2.shape)

outputs1 = model1(
    input_ids=combined_inputs1["input_ids"],
    attention_mask=combined_inputs1["attention_mask"],
    labels=labels1,
)
loss1 = outputs1.loss
logit1 = outputs1.logits

outputs2 = model2(
    input_ids=combined_inputs2["input_ids"],
    attention_mask=combined_inputs2["attention_mask"],
    labels=labels2,
)
loss2 = outputs2.loss
logit2 = outputs2.logits

print(logit1.shape)
print(logit2.shape)

def pad_logits(student_logits, teacher_logits, padding_value=0):
    if len(student_logits.shape) == 3: student_logits = student_logits.squeeze(0)
    if len(teacher_logits.shape) == 3: teacher_logits = teacher_logits.squeeze(0)
    student_size, teacher_size = student_logits.size(-2), teacher_logits.size(-2)
    pad_size = abs(student_size - teacher_size)
    if student_size > teacher_size:
        teacher_logits = F.pad(teacher_logits, (0, 0, 0, pad_size), value=padding_value)
    elif student_size < teacher_size:
        student_logits = F.pad(student_logits, (0, 0, 0, pad_size), value=padding_value)
    return student_logits, teacher_logits

def KL_divergence_token_level(logits_S, logits_T, valid_mask, temperature=1.0):
    """
    logits_S, logits_T : (B, L, V)
    valid_mask         : (B, L), 유효 토큰은 1, 패딩 토큰은 0
    temperature        : distillation temperature
    """
    # (1) 소프트맥스 & 로그소프트맥스
    p_T     = F.softmax(logits_T / temperature, dim=-1)       # (B, L, V)
    log_p_T = F.log_softmax(logits_T / temperature, dim=-1)   # (B, L, V)
    log_p_S = F.log_softmax(logits_S / temperature, dim=-1)   # (B, L, V)

    # (2) KL = Σ p_T * (log_p_T - log_p_S)
    kl_per_token = p_T * (log_p_T - log_p_S)  # (B, L, V)
    kl_per_token = kl_per_token.sum(dim=-1)   # (B, L)

    print(valid_mask.shape)
    valid_mask = torch.zeros(valid_mask.shape[0]).to(device)
    # 6.7580, validmask 있어서 padding value는 제외하고진행
    # 6.7594 validmask 없어서 패딩값들 까지 다 합쳐서 KL divergence 수행
    # (3) 유효 위치만 골라서(마스크 곱) 합산
    kl_per_token = kl_per_token * valid_mask  # 패딩(0)은 0이 됨

    # (4) 유효 토큰 개수로 평균
    kl_loss = kl_per_token.sum() / (valid_mask.sum() + 1e-9)

    # KD 논문에서 T^2를 곱해주는 관행이 있음
    return kl_loss * (temperature ** 2)

# projection layer needs

with autocast():
    linear1 = torch.nn.Linear(model2.get_input_embeddings().num_embeddings, 4096, dtype=torch.float16).to(device)
    linear2 = torch.nn.Linear(4096, model1.get_input_embeddings().num_embeddings, dtype=torch.float16).to(device)
    logit2 = linear1(logit2)
    logit2 = linear2(logit2)

    print(logit1.shape)
    print(logit2.shape)


    l_S, l_T = pad_logits(logit1, logit2)
    mask_S = (l_S != 0).any(dim=-1).float()
    mask_T = (l_T != 0).any(dim=-1).float()
    valid_mask = mask_S * mask_T

    total_kd_loss = KL_divergence_token_level(l_S, l_T, valid_mask)

    print(l_S)
    print(l_T)
    print(total_kd_loss)
    print(loss1)
    print(loss2)
