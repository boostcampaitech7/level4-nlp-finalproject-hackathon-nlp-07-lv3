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
from dotenv import load_dotenv

import wandb
from config import Config
from dataset import SALMONNDataset
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from runner import Runner
from utils import setup_logger


def now():
    seoul_tz = pytz.timezone("Asia/Seoul")
    return datetime.datetime.now(seoul_tz).strftime("%Y%m%d%H%M")


def parse_args():
    parser = argparse.ArgumentParser(description="train parameters")
    parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
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

    return seed


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    envs = load_dotenv()
    date_wandb = job_id[4:12]

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    run_config["output_dir"] = "outputs_stage_1"

    model_config = cfg.config.model
    data_config = cfg.config.datasets
    wandb_config = cfg.config.wandb

    # exp_name = input("write wandb exp_name:")
    # while exp_name == "":
    #     exp_name = input("write wandb exp_name:")
    exp_name = 'test'

    # Wandb setup
    if wandb_config.log:
        wandb.login(key=os.environ['WANDB_KEY'])
        wandb.init(
            project=wandb_config.project, entity=wandb_config.entity, name=date_wandb + "_Stage1_" + exp_name, config=cfg
        )

    # initialize distributed training
    init_distributed_mode(run_config)
    SEED = setup_seeds(run_config)
    setup_logger()  # set after init_distributed_mode() to only log on master.

    if run_config.use_distributed:  # 분산 모드 여부 확인
        global_rank = int(os.environ["RANK"])
    else:
        global_rank = 0

    print(f"Global rank: {global_rank}")

    # print config
    cfg.pretty_print()

    # build datasets
    if data_config.stage1_use_valid:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_stage1_path, model_config.whisper_path),
            "valid": SALMONNDataset(data_config.prefix, data_config.valid_stage1_path, model_config.whisper_path),
        }
    else:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_stage1_path, model_config.whisper_path),
        }

    # build model
    model = load_model(model_config)

    # build stage1 runner
    runner = Runner(cfg, model, datasets, job_id, args.dryrun, SEED)

    # stage1 train, return 마지막 ckpt 경로 넘겨 받음
    runner.train()

    # stage1 wandb 종료
    wandb.finish()

    if run_config.auto_second:
        del runner
        assert run_config["output_dir"]
        run_config["output_dir"] = "outputs_stage_2"

        if wandb_config.log:
            wandb.init(
                project=wandb_config.project, entity=wandb_config.entity, name=date_wandb + "_Stage2_" + exp_name, config=cfg
            )

        if data_config.stage2_use_valid:
            datasets = {
                "train": SALMONNDataset(data_config.prefix, data_config.train_stage2_path, model_config.whisper_path),
                "valid": SALMONNDataset(data_config.prefix, data_config.valid_stage2_path, model_config.whisper_path),
            }
        else:
            datasets = {
                "train": SALMONNDataset(data_config.prefix, data_config.train_stage2_path, model_config.whisper_path),
            }

        # build stage1 runner
        runner = Runner(cfg, model, datasets, job_id, args.dryrun, SEED)

        # stage1 train, return 마지막 ckpt 경로 넘겨 받음
        runner.train()

        # stage1 wandb 종료
        wandb.finish()


if __name__ == "__main__":
    main()
