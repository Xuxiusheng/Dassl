import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerBase, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.modeling.backbone import load_clip, clip, StyleGenerator
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

@TRAINER_REGISTRY.register()
class StyleTrainer(SimpleTrainer):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.cfg = cfg
        self.n_style = cfg.TRAINER.PROMPTSTYLER.N_STYLE
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, "styler")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.weight_path = os.path.join(self.output_dir, cfg.TRAINER.PROMPTSTYLER.CHECKPOINT_NAME)
        self.build_model()
    
    def build_model(self):
        cfg = self.cfg
        with open(cfg.TRAINER.PROMPTSTYLER.CLASS_DIR, 'r') as f:
            lines = f.readlines()
        self.classnames = [" ".join(line.strip().split("_")) for line in lines]
        self.model = StyleGenerator(cfg, self.classnames).cuda()
        for name, param in self.model.named_parameters():
            if "clip" in name:
                param.requires_grad_(False)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM, epochs=self.n_style*self.max_epoch)

    def train(self):
        start_time = time.time()
        self.model.train()
        for style_idx in range(self.n_style):
            for epoch in range(self.max_epoch):
                self.optim.zero_grad()
                content_consistency_loss, style_diversity_loss = self.model(style_idx)
                loss = style_diversity_loss + content_consistency_loss
                loss.backward()
                self.optim.step()
                self.sched.step()

                if (epoch + 1) % 20 == 0 or epoch == 0:
                    current_lr = self.optim.param_groups[0]["lr"]
                    info = []
                    info += [f"style_idx {style_idx}"]
                    info += [f"epoch [{epoch + 1}/{self.max_epoch}]"]
                    info += [f"loss {loss.item()}"]
                    info += [f"style_diversity_loss {style_diversity_loss.item()}"]
                    info += [f"content_consistency_loss {content_consistency_loss.item()}"]
                    info += [f"lr {current_lr:.4e}"]
                    print(" ".join(info))
        self.model.eval()
        torch.save(self.model.style_embedding, self.weight_path)
        print("-" * 20)
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        print("********finished********")

    

