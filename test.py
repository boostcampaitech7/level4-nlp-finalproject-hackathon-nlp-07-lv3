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

def pad_logits(student_logits, teacher_logits, padding_value=-2):
    if len(student_logits.shape) == 3: student_logits = student_logits.squeeze(0)
    if len(teacher_logits.shape) == 3: teacher_logits = teacher_logits.squeeze(0)
    student_size, teacher_size = student_logits.size(-2), teacher_logits.size(-2)
    pad_size = abs(student_size - teacher_size)
    if student_size > teacher_size:
        teacher_logits = F.pad(teacher_logits, (0, 0, 0, pad_size), value=padding_value)
    elif student_size < teacher_size:
        student_logits = F.pad(student_logits, (0, 0, 0, pad_size), value=padding_value)
    return student_logits, teacher_logits

tensor1 = torch.randn(1, 8, 4)
tensor2  = torch.randn(1, 4, 4)

tensor1, tensor2 = pad_logits(tensor1, tensor2)

print(tensor1)
print(tensor2)
