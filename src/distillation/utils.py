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

class DistillDataCollatorForSeq2Seq:

    Teachertokenizer: PreTrainedTokenizerBase
    Studenttokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.Teachertokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        teacher_features = self.Teachertokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        student_features = self.Studenttokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            teacher_features["decoder_input_ids"] = decoder_input_ids
            student_features["decoder_input_ids"] = decoder_input_ids

        batched_data = {
            'teacher': teacher_features,
            'student': student_features
        }

        return batched_data

def dynamic_kd_loss(student_logits, teacher_logits, temperature=1.0):
    with torch.no_grad():   
        student_probs = F.softmax(student_logits, dim=-1)
        student_entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=-1) # student entropy, (bsz, )
        # normalized entropy score by student uncertainty:
        # i.e.,  entropy / entropy_upper_bound
        # higher uncertainty indicates the student is more confusing about this instance
        instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))
    student_input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits / temperature, dim=-1)
    batch_loss = F.kl_div(student_input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
    weighted_kld = torch.mean(batch_loss * instance_weight)
    return weighted_kld

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

