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

file = load_file("/data/pgt/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/data/inf_result/meta-llama-Llama-3.2-3B-Instruct/stage1_train/meta-llama-Llama-3.2-3B-Instruct_stage1_train_0.safetensors")
print(file)