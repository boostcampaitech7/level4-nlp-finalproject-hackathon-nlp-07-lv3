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
from config import DistillConfig
from dataset import SALMONNDataset
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from textbrewer import TrainingConfig, DistillationConfig
from utils import setup_logger
from distill_runner import DistillRunner
from distillation import CustomDistiller, CustomDistiller2, CustomDistiller3

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
    date_wandb = job_id[4:12]

    # load config
    args = parse_args()
    cfg = DistillConfig(args)

    # stage1, stage2 각각의 optim과 output_dir을 따로 담아두고 나중에 넣어줌
    optims_1 = cfg.config.run.optims.optims_1
    output_dir_1 = cfg.config.run.output_dir.stage1_output_dir
    optims_2 = cfg.config.run.optims.optims_2
    output_dir_2 = cfg.config.run.output_dir.stage2_output_dir

    # stage1 optim 설정으로 바꾸기
    cfg.config.run.optims = optims_1
    cfg.config.run.output_dir = output_dir_1


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
        datasets2 = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_1, data_config.whisper_path2),
            "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path_1, data_config.whisper_path2),
        }

    else:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_1, data_config.whisper_path),
        }
        datasets2 = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_1, data_config.whisper_path2),
        }



    def simple_adaptor(model_outputs, batch=None):
        return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states, 'loss': model_outputs.loss}

    def simple_adaptor2(model_outputs, batch=None):
        return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states, 'loss': model_outputs.loss, 'last_hidden_state': model_outputs.hidden_states[-1]}

    # build model
    if not args.dryrun:
        if model_T_config.is_output_saved:
            model_T = None
        else:
            model_T = load_model(model_T_config)
        model_S = load_model(model_S_config)

        distiller = CustomDistiller3(
            adaptor_T=simple_adaptor2,
            adaptor_S=simple_adaptor2,
            qformer_dim_T=model_T.speech_Qformer.config.hidden_size,
            qformer_dim_S=model_S.speech_Qformer.config.hidden_size,
            logits_pro=['linear', model_S.llama_model.get_input_embeddings().num_embeddings, model_T.llama_model.get_input_embeddings().num_embeddings],
            student_device=run_config.device
        )

        # distiller = CustomDistiller2(
        #     adaptor_T=simple_adaptor2,
        #     adaptor_S=simple_adaptor2
        # )

        # distiller = CustomDistiller(
        #                 train_config=TrainingConfig(
        #                     device=run_config.device
        #                 ),
        #                 distill_config=DistillationConfig(),
        #                 model_T=model_T,
        #                 model_S=model_S,
        #                 adaptor_T=simple_adaptor,
        #                 adaptor_S=simple_adaptor,
        #                 logits_pro=['linear', model_S.llama_model.get_input_embeddings().num_embeddings, model_T.llama_model.get_input_embeddings().num_embeddings],
        #                 global_step_start=0,
        #                 use_softmax=True,
        #                 dt_normalization_type='softmax',
        #             )

    else:  # load small dummy language model
        return

    # build stage1 runner
    runner_1 = DistillRunner(cfg, model_T, model_S, distiller, datasets, datasets2, job_id, args.dryrun, SEED, is_custom_3=True)

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
    #     datasets2 = {
    #         "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_2, data_config.whisper_path2),
    #         "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path_2, data_config.whisper_path2),
    #     }

    # else:
    #     datasets = {
    #         "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_2, data_config.whisper_path),
    #     }
    #     datasets2 = {
    #         "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_2, data_config.whisper_path2),
    #     }

    # # stage2 optim 설정으로 바꾸기
    # cfg.config.run.optims = optims_2
    # cfg.config.run.output_dir = output_dir_2
    # cfg.config.model_S.ckpt = ckpt_path

    # if not args.dryrun:
    #     model_S = load_model(model_S_config)

    # # print config
    # cfg.pretty_print()

    # # Wandb setup, stage2 wandb 시작
    # if wandb_config.log:
    #     wandb.init(
    #         project=wandb_config.project, entity=wandb_config.entity, name=date_wandb + "_AAC_", config=cfg
    #     )

    # # build stage2 runner
    # runner_2 = DistillRunner(cfg, model_T, model_S, distiller, datasets, datasets2, job_id, args.dryrun, SEED)

    # # stage2 train
    # runner_2.train()

if __name__ == "__main__":
    main()
