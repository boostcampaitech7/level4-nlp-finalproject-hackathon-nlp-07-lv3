from typing import Optional, Any, Union
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file

def softmax_normalize(tensor, dim=-1):
    return F.softmax(tensor, dim=dim)

def minmax_normalize(tensor, dim=-1):
    min_vals, _ = torch.min(tensor, dim=dim, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=dim, keepdim=True)

    epsilon = 1e-8
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + epsilon)

    return normalized_tensor

def standardize_tensor(tensor, dim=-1):
    mean_vals = torch.mean(tensor, dim=dim, keepdim=True)
    std_vals = torch.std(tensor, dim=dim, keepdim=True)

    epsilon = 1e-8
    standardized_tensor = (tensor - mean_vals) / (std_vals + epsilon)

    return standardized_tensor

def pad_logits(student_logits, teacher_logits, padding_value=-100):
    if len(student_logits.shape) == 3: student_logits = student_logits.squeeze(0)
    if len(teacher_logits.shape) == 3: teacher_logits = teacher_logits.squeeze(0)
    student_size, teacher_size = student_logits.size(-2), teacher_logits.size(-2)
    pad_size = abs(student_size - teacher_size)
    if student_size > teacher_size:
        teacher_logits = F.pad(teacher_logits, (0, 0, 0, pad_size))
    elif student_size < teacher_size:
        student_logits = F.pad(student_logits, (0, 0, 0, pad_size))
    return student_logits, teacher_logits

def read_teacher_outputs(teacher_output_path: str, device='cuda'):
    loaded_data = load_file(teacher_output_path)
    if "logits" not in loaded_data:
        raise KeyError(f"'logits' key not found in {teacher_output_path}")
    
    results = {
        "logits": loaded_data["logits"].to(device),
    }
    if "loss" in loaded_data:
        results["loss"] = loaded_data["loss"].to(device)
    
    return results

def custom_post_adaptor(dict_object):
    if 'logits' in dict_object:
        logits = dict_object['logits']
        if not isinstance(logits,(list,tuple)):
            dict_object['logits'] = [ logits ]
    if 'logits_mask' in dict_object:
        logits_mask = dict_object['logits_mask']
        if not isinstance(logits_mask,(list,tuple)):
            dict_object['logits_mask'] = [ logits_mask ]
    if 'loss' in dict_object:
        loss = dict_object['loss']
        if not isinstance(loss,(list,tuple)):
            dict_object['loss'] = [ loss ]
    if 'labels' in dict_object:
        labels = dict_object['labels']
        if not isinstance(labels,(list,tuple)):
            dict_object['labels'] = [ labels ]
    if 'embeds' in dict_object:
        embeds = dict_object['embeds']
        if not isinstance(embeds,(list,tuple)):
            dict_object['embeds'] = [ embeds ]
    return dict_object

class CustomDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'CustomDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"'CustomDict' object has no attribute '{name}'")