import random
from typing import Optional, Any, Union
import itertools
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import sys

sys.path.append('./src/lm_evaluation_harness')
from lm_eval import tasks, evaluator
import lm_eval
import json
import logging
import fnmatch
import collections
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Softmax, CrossEntropyLoss
from safetensors.torch import save_file, load_file

from textbrewer import GeneralDistiller
from textbrewer.distiller_utils import *
from textbrewer.distiller_basic import BasicDistiller

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

def dynamic_temperature(student_logits, teacher_logits, normalization_type=''):
    if len(normalization_type)>0:
        if normalization_type=='minmax':
            student_logits = minmax_normalize(student_logits)
            teacher_logits = minmax_normalize(teacher_logits)
        elif normalization_type=='softmax':
            student_logits = softmax_normalize(student_logits)
            teacher_logits = softmax_normalize(teacher_logits)
        elif normalization_type == 'standardize':
            student_logits = standardize_tensor(student_logits)
            teacher_logits = standardize_tensor(teacher_logits)

    tea_std = torch.std(teacher_logits, dim=-1,keepdim=True)
    stu_std= torch.std(student_logits, dim=-1, keepdim=True)
    p_s = F.log_softmax(student_logits/tea_std, dim=1)
    p_t = F.softmax(teacher_logits/stu_std, dim=1)
    # pdb.set_trace()
    loss = torch.sum(torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=-1) * (1 * torch.ones(student_logits.shape[0],1).cuda())) /student_logits.shape[0]/ student_logits.shape[0]
    return loss

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

def read_teacher_outputs(teacher_output_path: str):
    loaded_data = load_file(teacher_output_path)
    return loaded_data

def custom_post_adaptor(dict_object):
    if 'logits' in dict_object:
        logits = dict_object['logits']
        if not isinstance(logits,(list,tuple)):
            dict_object['logits'] = [ logits ]
    if 'logits_mask' in dict_object:
        logits_mask = dict_object['logits_mask']
        if not isinstance(logits_mask,(list,tuple)):
            dict_object['logits_mask'] = [ logits_mask ]
    if 'losses' in dict_object:
        losses = dict_object['losses']
        if not isinstance(losses,(list,tuple)):
            dict_object['losses'] = [ losses ]
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