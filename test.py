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

tensor = torch.randn(6, 3, 4)
dic = {}
dic['tensor'] = [tensor]

for val in dic['tensor']:
    print(val.shape)