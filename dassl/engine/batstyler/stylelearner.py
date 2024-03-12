import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerBase, SimpleTrainer
from collections import OrderedDict
import os
from dassl.modeling.backbone import ETFHead
from dassl.optim import build_optimizer, build_lr_scheduler
import time
import datetime

@TRAINER_REGISTRY.register()
class PromptLearnerTrainer(SimpleTrainer):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.cfg = cfg
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, "learner")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.weight_path = os.path.join(self.output_dir, cfg.TRAINER.BATSTYLER.CHECKPOINT_NAME)
        self.build_model()
        self.n_style = cfg.TRAINER.BATSTYLER.N_STYLE

    def build_model(self):
        cfg = self.cfg
        with open(cfg.TRAINER.BATSTYLER.CLASS_DIR, 'r') as f:
            lines = f.readlines()
        self.classnames = [" ".join(line.strip().split("_")).lower() for line in lines]
        self.model = ETFHead(cfg, self.classnames).cuda()
        for name, param in self.model.named_parameters():
            if "clip" in name:
                param.requires_grad_(False)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim)

        

    def train(self):
        start_time = time.time()
        self.model.train()
        for epoch in range(self.max_epoch):
            self.optim.zero_grad()
            output = self.model()
            loss = self.model.loss(output)
            loss.backward()
            self.optim.step()
            self.sched.step()

            if (epoch + 1) % 20 == 0:
                current_lr = self.optim.param_groups[0]["lr"]
                info = []
                info += [f"epoch [{epoch + 1}/{self.max_epoch}]"]
                info += [f"loss {loss.item()}"]
                info += [f"lr {current_lr:.4e}"]
                print(" ".join(info))

        self.model.eval()
        torch.save(self.model.style_embedding, self.weight_path)
        print("-" * 20)
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        print("********finished********")