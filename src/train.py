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
from runner import Runner
from utils import setup_logger


def now():
    seoul_tz = pytz.timezone("Asia/Seoul")
    return datetime.datetime.now(seoul_tz).strftime("%Y%m%d%H%M")


def parse_args():
    parser = argparse.ArgumentParser(description="train parameters")
    parser.add_argument("--cfg-path", type=str, required=True, help="path to configuration file")
    parser.add_argument("--device", type=str, default="cuda:0")
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
    date_wandb = job_id[4:12]

    # load config
    args = parse_args()
    cfg = Config(args)

    # stage1, stage2 각각의 optim과 output_dir을 따로 담아두고 나중에 넣어줌
    optims_1 = cfg.config.run.optims.optims_1
    output_dir_1 = cfg.config.run.output_dir.stage1_output_dir
    optims_2 = cfg.config.run.optims.optims_2
    output_dir_2 = cfg.config.run.output_dir.stage2_output_dir

    # stage1 optim 설정으로 바꾸기
    cfg.config.run.optims = optims_1
    cfg.config.run.output_dir = output_dir_1

    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets
    wandb_config = cfg.config.wandb

    exp_name = input("write wandb exp_name:")

    # Wandb setup
    if wandb_config.log:
        wandb.login(key=wandb_config.key)
        wandb.init(
            project=wandb_config.project, entity=wandb_config.entity, name=date_wandb + "_ASR_" + exp_name, config=cfg
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

    # build stage1 datasets
    # 별도로 valid 지정 없는 경우 train만 생성 후 split
    if data_config.valid_ann_path_1:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_1, data_config.whisper_path),
            "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path_1, data_config.whisper_path),
        }

    else:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_1, data_config.whisper_path),
        }

    # build model
    if not args.dryrun:
        model = load_model(model_config)
    else:  # load small dummy language model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)

    # build stage1 runner
    runner_1 = Runner(cfg, model, datasets, job_id, args.dryrun, SEED)

    # stage1 train, return 마지막 ckpt 경로 넘겨 받음
    ckpt_path = runner_1.train()

    # stage1 wandb 종료
    wandb.finish()

    # # build stage2 datasets
    # # 별도로 valid 지정 없는 경우 train만 생성 후 split
    # if data_config.valid_ann_path_2:
    #     datasets = {
    #         "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_2, data_config.whisper_path),
    #         "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path_2, data_config.whisper_path),
    #     }

    # else:
    #     datasets = {
    #         "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_2, data_config.whisper_path),
    #     }

    # # stage2 optim 설정으로 바꾸기
    # cfg.config.run.optims = optims_2
    # cfg.config.run.output_dir = output_dir_2
    # cfg.config.model.ckpt = ckpt_path

    # # print config
    # cfg.pretty_print()

    # # Wandb setup, stage2 wandb 시작
    # if wandb_config.log:
    #     wandb.init(
    #         project=wandb_config.project, entity=wandb_config.entity, name=date_wandb + "_AAC_" + exp_name, config=cfg
    #     )

    # # build stage2 runner
    # runner_2 = Runner(cfg, model, datasets, job_id, args.dryrun, SEED)

    # # stage2 train
    # runner_2.train()


if __name__ == "__main__":
    main()
