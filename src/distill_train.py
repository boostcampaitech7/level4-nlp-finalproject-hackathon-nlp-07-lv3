# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import os
import random

import numpy as np
import pytz
import torch
import torch.backends.cudnn as cudnn

import wandb
from config import Config
from dataset import SALMONNDataset
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from distillation import CustomDistiller
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
from utils import setup_logger
from distill_runner import DistillRunner

def now():
    seoul_tz = pytz.timezone("Asia/Seoul")
    return datetime.datetime.now(seoul_tz).strftime("%Y%m%d%H%M")


def parse_args():
    parser = argparse.ArgumentParser(description="train parameters")
    parser.add_argument("--cfg-model-T-path", type=str, required=True, help="path to configuration file")
    parser.add_argument("--cfg-model-S-path", type=str, required=True, help="path to configuration file")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--dryrun", action="store_true", help="if True, use dummy model and skip forward/backward")

    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def simple_adaptor(batch, model_outputs):
    return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states, 'losses': model_outputs.loss}

def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    args = parse_args()
    cfg = DistillConfig(args)
    run_config = cfg.config.run
    model_T_config = cfg.config.model_T
    model_S_config = cfg.config.model_S
    data_config = cfg.config.datasets
    wandb_config = cfg.config.wandb

    # Wandb setup
    if wandb_config.log:
        wandb.login(key=wandb_config.key)
        wandb.init(project=wandb_config.project, entity=wandb_config.entity, config=cfg)

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger()  # set after init_distributed_mode() to only log on master.

    if run_config.use_distributed:  # 분산 모드 여부 확인
        global_rank = int(os.environ["RANK"])
    else:
        global_rank = 0

    print(f"Global rank: {global_rank}")

    # print config
    cfg.pretty_print()

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        # "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        # "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }

    # build model
    if not args.dryrun:
        model_T = load_model(model_T_config)
        model_S = load_model(model_S_config)
    else:  
        return

    # build runner
    runner = DistillRunner(cfg, model_T, model_S, datasets, job_id, args.dryrun)

    # train
    runner.train()

if __name__ == "__main__":
    main()
