import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerBase, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.modeling.backbone import load_clip, clip, Classifier
from dassl.optim import build_optimizer, build_lr_scheduler
from collections import OrderedDict
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import datetime
import time
import torch.nn.functional as F
import os
from dassl.ops import OurStyleGenerator
from dassl.data import DataManager
from tqdm import tqdm
import numpy as np

@TRAINER_REGISTRY.register()
class PromptStylerLPTrainer(SimpleTrainer):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.cfg = cfg
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, "lp")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.build_model()
        self.build_data_loader()
    
    def build_model(self):
        cfg = self.cfg
        with open(cfg.TRAINER.PROMPTSTYLER.CLASS_DIR, 'r') as f:
            lines = f.readlines()
        self.classnames = [line.strip() for line in lines]
        self.model = Classifier(cfg, self.classnames).cuda()
        for name, param in self.model.named_parameters():
            if "clip" in name:
                param.requires_grad_(False)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
    
    def build_data_loader(self):
        cfg = self.cfg
        train_data = OurStyleGenerator(cfg, self.classnames, self.model.clip_model).train_data()
        dm = DataManager(self.cfg, train_data)
        self.train_loader_x = dm.train_loader_x
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm
    
    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            if meet_freq:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

        n_iter = self.epoch * self.num_batches + self.batch_idx
        for name, meter in losses.meters.items():
            self.write_scalar("train/" + name, meter.avg, n_iter)
        self.write_scalar("train/lr", self.get_current_lr(), n_iter)

        end = time.time()

    def parse_batch_train(self, batch):
        prompt, tokenized_text, label = batch["prompt"].cuda(), batch["tokenized_text"].cuda(), batch["label"].cuda()
        return prompt, tokenized_text, label
    
    def forward_backward(self, batch):
        prompt, tokenized_text, label = self.parse_batch_train(batch)
        output = self.model(prompt, tokenized_text)
        output = F.normalize(output, dim=1)
        loss, y_pred = self.model.arcface_loss(output, label)
        self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(y_pred, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary
    
    def after_epoch(self):
        result, is_best = self.test()
        if is_best:
            self.save_model(
                self.epoch,
                self.output_dir,
                val_result=result,
                model_name="model-best.pth"
            )
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        result = []
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        for data_loader_domain in data_loader:
            print(f"Evaluate on the *{split}* set")
            for batch_idx, batch in enumerate(tqdm(data_loader_domain)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, label)

            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
            result.append(list(results.values())[0])
            self.evaluator.reset()
        mean_acc = np.mean(result)
        print("-" * 10)
        print(f"Mean-Acc:{mean_acc}")
        print("*" * 10)
        is_best = False
        if self.best_result[-1] < mean_acc:
            self.best_result = (result, mean_acc)
            is_best = True
        return result, is_best
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.cuda()
        label = label.cuda()
        return input, label
    
    def model_inference(self, input):
        return self.model.forward_img(input)