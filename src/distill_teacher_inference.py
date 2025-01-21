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
from safetensors.torch import save_file
from torch.utils.data import random_split

from config import DistillConfig, Config
from dataset import SALMONNDataset
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from utils import setup_logger
from distill_runner import DistillRunner
from utils import get_dataloader, prepare_sample

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

    # Wandb setup
    if wandb_config.log:
        wandb.login(key=wandb_config.key)
        wandb.init(project=wandb_config.project, entity=wandb_config.entity, config=cfg)

    # initialize distributed training
    init_distributed_mode(run_config)
    SEED = setup_seeds(run_config)
    device = torch.device(cfg.config.run.device)

    setup_logger()  # set after init_distributed_mode() to only log on master.

    if run_config.use_distributed:  # 분산 모드 여부 확인
        global_rank = int(os.environ["RANK"])
    else:
        global_rank = 0

    print(f"Global rank: {global_rank}")

    
    # print config
    cfg.pretty_print()

    # build stage1 datasets
    if data_config.valid_ann_path_1:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_1, data_config.whisper_path),
            "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path_1, data_config.whisper_path),
        }

    else:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_1, data_config.whisper_path),
        }

    train_dataset = datasets["train"]

    dataloader = get_dataloader(
              train_dataset, cfg.config.run, is_train=False, use_distributed=False
            )
    # build model
    if not args.dryrun:
        model = load_model(model_config)
        model.to(device).eval()
    else:  # load small dummy language model
        return

    lm_path = model_config.llama_path.replace("/", "-")
    stage_path = data_config.train_ann_path_1.split("/")[-1].split(".")[0]
    output_dir = f"/data/pgt/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/data/inf_result/{lm_path}/{stage_path}" 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # build stage1 runner
    for i, samples in tqdm(enumerate(dataloader),desc="[Inferencing]", total=len(dataloader)):
        samples = prepare_sample(samples, cuda_enabled=True, device=device)
        with torch.cuda.amp.autocast():
            outputs = model(samples)    

        # hidden_states = outputs.hidden_states  # 모든 레이어의 hidden states (tuple 형태)
        logits = outputs.logits 
        loss = outputs.loss if "loss" in outputs else None  


        data_to_save = {
            "logits": logits.cpu(),
            # "hidden_states_last_layer": hidden_states[-1].cpu(),  # 마지막 레이어의 hidden states
        }


        if loss is not None:
            data_to_save["loss"] = loss.cpu()

        save_file(data_to_save, os.path.join(output_dir, f"{lm_path}_{stage_path}_{i}.safetensors"))

    # stage1 wandb 종료
    wandb.finish()

    # build stage2 datasets
    # 별도로 valid 지정 없는 경우 train만 생성 후 split
    if data_config.valid_ann_path_2:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_2, data_config.whisper_path),
            "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path_2, data_config.whisper_path),
        }

    else:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path_2, data_config.whisper_path),
        }


    train_dataset = datasets["train"]
    dataloader = get_dataloader(
              train_dataset, cfg.config.run, is_train=False, use_distributed=False
            )

    # stage2 optim 설정으로 바꾸기
    cfg.config.run.optims = optims_2
    cfg.config.run.output_dir = output_dir_2

    cfg.pretty_print()

    if wandb_config.log:
        wandb.init(
            project=wandb_config.project, entity=wandb_config.entity, name=date_wandb + "_AAC_", config=cfg
        )

     
    lm_path = model_config.llama_path.replace("/", "-")
    stage_path = data_config.train_ann_path_2.split("/")[-1].split(".")[0]
    output_dir = f"/data/pgt/level4-nlp-finalproject-hackathon-nlp-07-lv3/src/data/inf_result/{lm_path}/{stage_path}" 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i, samples in tqdm(enumerate(dataloader),desc="[Inferencing]", total=len(dataloader)):
        samples = prepare_sample(samples, cuda_enabled=True, device=device)
        with torch.cuda.amp.autocast():
            outputs = model(samples)

        logits = outputs.logits 
        loss = outputs.loss if "loss" in outputs else None 

        data_to_save = {
            "logits": logits.cpu(),
            # "hidden_states_last_layer": hidden_states[-1].cpu(),  # 마지막 레이어의 hidden states
        }

        if loss is not None:
            data_to_save["loss"] = loss.cpu()

        save_file(data_to_save, os.path.join(output_dir, f"{lm_path}_{stage_path}_{i}.safetensors"))

if __name__ == "__main__":
    main()
