# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import copy
import datetime
import glob
import json
import logging
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
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
from models.json_to_manifest import json_to_manifest_indice, json_to_manifest
from nemo.collections.asr.models import EncDecMultiTaskModel
from models.modeling_canary import get_dataloader_from_config

class Runner:
    def __init__(self, cfg, model, datasets, job_id, dryrun, SEED):
        self.seed = SEED
        self.config = cfg

        # dryrun (test with dummy model)
        self.dryrun = dryrun

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        #self.device = torch.device(self.config.config.run.device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_distributed = self.config.config.run.use_distributed
        self.start_epoch = 0
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.evaluate_only = self.config.config.run.evaluate
        self.cuda_enabled = self.device.type == "cuda"

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r", encoding="utf-8") as f:
                    self.test_prompt_dict = json.load(f)
            except json.JSONDecodeError:
                print("Failed to decode JSON! Trying with utf-8 encoding.")
                try:
                    with open(test_prompt_path, "r", encoding="utf-8") as f:
                        self.test_prompt_dict = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error even with utf-8 encoding: {e}")
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

        if "valid" in datasets.keys():
            train_dataset = datasets["train"]
            _ = json_to_manifest(self.config.config.datasets.train_ann_path_1, self.config.config.datasets.train_manifest_path)

            valid_dataset = datasets["valid"]
            _ = json_to_manifest(self.config.config.datasets.valid_ann_path_1, self.config.config.datasets.valid_manifest_path)

        else:
            train_dataset = datasets["train"]

            train_size = int(0.95 * len(train_dataset))
            valid_size = len(train_dataset) - train_size

            train_indices, valid_indices = random_split(
                range(len(train_dataset)), [train_size, valid_size], generator=torch.Generator().manual_seed(self.seed)
            )

            valid_dataset = copy.deepcopy(train_dataset)
            train_dataset.annotation = [train_dataset.annotation[i] for i in train_indices]
            valid_dataset.annotation = [valid_dataset.annotation[i] for i in valid_indices]

            # make temporary manifest file for train and validations
            assert(self.config.config.datasets.train_manifest_path != '')
            assert(self.config.config.datasets.valid_manifest_path != '')

            _ = json_to_manifest_indice(self.config.config.datasets.train_ann_path_1, self.config.config.datasets.train_manifest_path, train_indices)
            _ = json_to_manifest_indice(self.config.config.datasets.train_ann_path_1, self.config.config.datasets.valid_manifest_path, valid_indices)

        test_dataset = datasets["test"] if "test" in datasets else None

        # 데이터로더 생성
        self.train_loader = get_dataloader(
            train_dataset, self.config.config.run, is_train=True, use_distributed=self.use_distributed
        )
        self.valid_loader = get_dataloader(
            valid_dataset, self.config.config.run, is_train=False, use_distributed=self.use_distributed
        )
        self.test_loader = get_dataloader(
            test_dataset, self.config.config.run, is_train=False, use_distributed=self.use_distributed
        ) if test_dataset else None

        self.train_config, self.n_loader_train = get_dataloader_from_config(self.model.speech_encoder, self.config.config.datasets.train_manifest_path, batch_size=self.config.config.run.batch_size_train)
        self.validation_config, self.n_loader_valid = get_dataloader_from_config(self.model.speech_encoder, self.config.config.datasets.valid_manifest_path, batch_size=self.config.config.run.batch_size_eval)

        # self.n_loader_train.batch_size = 4
        # self.n_loader_valid.batch_size = 4

        self.model.speech_encoder._update_dataset_config(dataset_name='train', config=self.train_config)
        self.model.speech_encoder._train_dl = self.n_loader_train

        self.model.speech_encoder._update_dataset_config(dataset_name='validation', config=self.validation_config)
        self.model.speech_encoder._validation_dl = self.n_loader_valid

        # scaler
        self.use_amp = self.config.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
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

    def train_epoch(self, epoch, profile_flag=False):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info("Start training epoch {}, {} iters per inner epoch.".format(epoch, self.iters_per_epoch))
        header = "Train: data epoch: [{}]".format(epoch)

        for i, n_samples in zip(metric_logger.log_every(range(self.iters_per_epoch), self.config.config.run.log_freq, header=header, logger=self.log_writter, start_step=epoch * self.iters_per_epoch,),
                                      self.n_loader_train._get_iterator()):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            if not self.dryrun:
                self.scheduler.step(cur_epoch=epoch, cur_step=i)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    profile_flag = True if i == 0 and profile_flag else False
                    loss = self.model(samples, n_samples, profile_flag=profile_flag)["loss"]

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
            else:  # dryrun, no model availble
                metric_logger.update(loss=0.0)
                metric_logger.update(lr=0.0)
                global_rank = int(os.environ["RANK"])
                if global_rank == 0:
                    wandb.log({"train/iteration": i, "train/loss": 0.0, "train/lr": 0.0})

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def valid_epoch(self, epoch, split, decode=False, save_json=False):
        if not self.dryrun:
            model = self.unwrap_dist_model(self.model)
            model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval: data epoch: [{}]".format(epoch)

        results = []
        for samples, n_samples in zip(metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header), self.n_loader_valid._get_iterator()):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            if not self.dryrun:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    forward_result = model(samples, n_samples, verbose=True)
                loss = forward_result.get("loss", 0)
                correct = forward_result.get("correct", 0)
                total = forward_result.get("total", 1)
                res = {
                    "id": samples["id"],
                    "ground_truth": samples["text"],
                    "loss": loss.item(),
                    "acc": (correct / total).item(),
                    "total": total,
                }
            else:
                res = {
                    "id": samples["id"],
                    "ground_truth": samples["text"],
                    "loss": 0.0,
                    "acc": 0.0,
                    "total": 1,
                }

            if decode:
                if model.prompt_dict:
                    if self.test_prompt_dict is None:
                        prompts = None
                    else:
                        prompts = [self.test_prompt_dict[s] for s in samples["task"]]
                        if "Q" in samples:
                            prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, samples["Q"])]
                else:
                    prompts = None

                text = model.generate(samples, self.config.config.run, prompts=prompts)
                res["text"] = text
                res["prompt"] = prompts
                res["task"] = samples["task"]

            results.append(res)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            self.save_result(results, self.output_dir, "eval_{}_epoch_{}".format(split, epoch))

        res = {
            "loss": torch.tensor(0).float().cuda(),
            "n_sample": torch.tensor(0).float().cuda(),
            "correct": torch.tensor(0).float().cuda(),
            "n_token": torch.tensor(0).float().cuda(),
        }

        for item in results:
            item_loss = item["loss"]
            item_n_sample = len(item["id"])
            item_correct = item["acc"] * item["total"]
            item_n_token = item["total"]
            res["loss"] += item_loss * item_n_sample
            res["n_sample"] += item_n_sample
            res["correct"] += item_correct
            res["n_token"] += item_n_token

        if is_dist_avail_and_initialized():
            dist.all_reduce(res["loss"])
            dist.all_reduce(res["n_sample"])
            dist.all_reduce(res["correct"])
            dist.all_reduce(res["n_token"])

        ret = {"loss": 0, "agg_metrics": 0}
        ret["loss"] = (res["loss"] / res["n_sample"]).item()
        ret["agg_metrics"] = (res["correct"] / res["n_token"]).item()

        return ret

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
        best_agg_metric = 0
        best_epoch = 0

        # testing phase
        start, mid, end = 0, (self.start_epoch + self.max_epoch - 1) // 2, self.max_epoch - 1
        save_directory = ""

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            logging.info("Training Phase")

            if cur_epoch == start or cur_epoch == mid or cur_epoch == end:
                train_stats = self.train_epoch(cur_epoch, profile_flag=True)

            else:
                train_stats = self.train_epoch(cur_epoch, profile_flag=False)

            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")

            # Test how much iteration self.n_loader_train has
            for elem in self.n_loader_valid._get_iterator():
                temp = elem
                print(temp.audio.shape)

            valid_log = self.valid_epoch(cur_epoch, "valid", decode=False, save_json=False)
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = valid_log["agg_metrics"]
                    if agg_metrics > best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch

                        save_directory = self.save_checkpoint(cur_epoch, is_best=True)

                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")
                    wandb.log({"valid/epoch": cur_epoch, "valid/agg_metrics": agg_metrics})

            if self.use_distributed:
                dist.barrier()

        save_directory = self.save_checkpoint(cur_epoch, is_best=False)

        if self.evaluate_only:
            test_log = self.valid_epoch("best", "test", decode=True, save_json=True)
            if test_log is not None:
                self.log_stats(test_log, split_name="test")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

        assert(save_directory != "")
        return save_directory

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
    def save_checkpoint(self, cur_epoch, is_best=False):
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
            save_to = os.path.join(self.output_dir, f"checkpoint_{cur_epoch}.pth")

        logging.info(f"Saving checkpoint at epoch {cur_epoch} to {save_to}.")
        torch.save(save_obj, save_to)

        # Keep only the most recent two checkpoints
        if not is_best:
            checkpoints = sorted(glob.glob(os.path.join(self.output_dir, "checkpoint_*.pth")))
            if len(checkpoints) > 2:
                oldest_checkpoint = checkpoints[0]
                os.remove(oldest_checkpoint)
                logging.info(f"Removed old checkpoint {oldest_checkpoint}.")

        return save_to
