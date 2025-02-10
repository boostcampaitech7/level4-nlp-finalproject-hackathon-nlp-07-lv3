# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py
import copy
import datetime
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split

import wandb
from dist_utils import get_rank, get_world_size, is_dist_avail_and_initialized, is_main_process, main_process
from logger import MetricLogger, SmoothedValue
from optims import LinearWarmupCosineLRScheduler, get_optimizer
from utils import get_dataloader, prepare_sample


class Runner:
    def __init__(self, cfg, model, datasets, job_id, dryrun, SEED):
        # SEED 설정
        self.seed = SEED

        self.config = cfg

        # dryrun (test with dummy model)
        self.dryrun = dryrun

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        self.device = torch.device(self.config.config.run.device)
        self.use_distributed = self.config.config.run.use_distributed
        self.start_epoch = 0
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.cuda_enabled = self.device.type == "cuda"

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r", encoding="utf-8") as f:
                    self.test_prompt_dict = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error with utf-8 encoding: {e}")
                raise
            except IOError as e:
                print(f"Failed to open or read the file: {e}")
                raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise

            for k, v in self.test_prompt_dict.items():
                self.test_prompt_dict[k] = self.prompt_template.format(v)

        else:
            self.test_prompt_dict = None

        # model
        self._model = model
        self._model.to(self.device)
        if self.use_distributed:
            self.model = DDP(self._model, device_ids=[self.config.config.run.gpu])
        else:
            self.model = self._model

        train_dataset = datasets["train"]

        # valid 있는 경우와 없는 경우로 나누어서 데이터셋 생성
        if "valid" in datasets.keys():
            print("valid가 있습니다.")
            valid_dataset = datasets["valid"]
        else:
            print("valid가 없으므로 train에서 9.5:0.5 비율로 생성합니다.")
            train_size = int(0.95 * len(train_dataset))
            valid_size = len(train_dataset) - train_size

            train_indices, valid_indices = random_split(
                range(len(train_dataset)), [train_size, valid_size], generator=torch.Generator().manual_seed(self.seed)
            )

            valid_dataset = copy.deepcopy(train_dataset)
            train_dataset.annotation = [train_dataset.annotation[i] for i in train_indices]
            valid_dataset.annotation = [valid_dataset.annotation[i] for i in valid_indices]

        # 데이터로더 생성
        self.train_loader = get_dataloader(
            train_dataset, self.config.config.run, is_train=True, use_distributed=self.use_distributed
        )
        self.valid_loader = get_dataloader(
            valid_dataset, self.config.config.run, is_train=False, use_distributed=self.use_distributed
        )

        # scaler
        self.use_amp = self.config.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # optimizer & scheduler
        self.iters_per_epoch = (
            len(self.train_loader) if self.config.config.run.epoch_based else self.config.config.run.iters_per_epoch
        )
        self.optimizer = get_optimizer(self.model, self.config.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.config.run.optims.min_lr,
            init_lr=self.config.config.run.optims.init_lr,
            warmup_steps=self.config.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.config.run.optims.get("warmup_start_lr", -1),
        )

        self.log_config()

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info("Start training epoch {}, {} iters per inner epoch.".format(epoch, self.iters_per_epoch))
        header = "Train: data epoch: [{}]".format(epoch)

        for i in metric_logger.log_every(
            range(self.iters_per_epoch),
            self.config.config.run.log_freq,
            header=header,
            logger=self.log_writter,
            start_step=epoch * self.iters_per_epoch,
        ):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled, device=self.config.config.run.device)

            if not self.dryrun:
                self.scheduler.step(cur_epoch=epoch, cur_step=i)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    loss = self.model(samples)["loss"]

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % self.config.config.run.accum_grad_iters == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

                global_rank = int(os.environ.get("RANK", 0))
                # global_rank = int(os.environ["RANK"]) 원래 코드

                if global_rank == 0:
                    wandb.log(
                        {
                            "train/iteration": i,
                            "train/loss": loss.item(),
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )
                # 1만 iter 마다 체크포인트 저장
                if i > 0 and i % 10000 == 0:
                    self.save_checkpoint(cur_epoch = epoch, iteration = i, is_best=False)
            else:  # dryrun, no model availble
                metric_logger.update(loss=0.0)
                metric_logger.update(lr=0.0)
                global_rank = int(os.environ["RANK"])
                if global_rank == 0:
                    wandb.log({"train/iteration": i, "train/loss": 0.0, "train/lr": 0.0})

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)

            print("result file saved to %s" % final_result_file)

    def train(self):
        start_time = time.time()
        best_save_directory = None  # 가장 좋은 모델 경로를 추적

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            if self.use_distributed:
                dist.barrier()

        # 가장 마지막 epoch의 모델은 val결과와 무관하게 저장
        last_save_directory = self.save_checkpoint(cur_epoch, iteration="last", is_best=False)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

        return best_save_directory if best_save_directory is not None else last_save_directory

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, iteration, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }

        if is_best:
            save_to = os.path.join(self.output_dir, "checkpoint_best.pth")
        else:
            save_to = os.path.join(self.output_dir, f"checkpoint_{cur_epoch}_{iteration}.pth")

        logging.info(f"Saving checkpoint at epoch {cur_epoch}_{iteration} to {save_to}.")
        torch.save(save_obj, save_to)

        # Keep only the most recent two checkpoints
        if not is_best:
            checkpoints = sorted(glob.glob(os.path.join(self.output_dir, "checkpoint_*.pth")))
            if len(checkpoints) > 2:
                oldest_checkpoint = checkpoints[0]
                os.remove(oldest_checkpoint)
                logging.info(f"Removed old checkpoint {oldest_checkpoint}.")

        return save_to


if __name__ == "__main__":
    subprocess.run([f"{sys.executable}", "train.py", "--cfg-path", "./configs/train.yaml"])
