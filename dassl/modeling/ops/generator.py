from torch import nn
import os
import torch
from dassl.modeling.backbone import clip
import random
import string

class OurStyleGenerator(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.PROMPTSTYLER.N_STYLE
        self.clip_model = clip_model
        self.weight_path = os.path.join(cfg.OUTPUT_DIR, "styler", cfg.TRAINER.PROMPTSTYLER.CHECKPOINT_NAME)
        self.base_text = "a X style of a"
        self.load_weight()

        for param in self.clip_model.parameters():
            param.requires_grad_(False)

    def load_weight(self):
        assert os.path.exists(self.weight_path), "prompt style weight path not exist!"
        self.style_embedding = torch.load(self.weight_path, map_location="cuda")
        self.style_embedding.requires_grad_(False)
        print(f"load weight from {self.weight_path}")

    def get_new_embedding(self, classname, idx):
        cfg = self.cfg
        sc_template = self.base_text + " " + classname
        current_style = self.style_embedding[idx:idx+1, :, :]
        with torch.no_grad():
            tokenized_sc = clip.tokenize(sc_template).cuda()
            sc_embedding = self.clip_model.token_embedding(tokenized_sc)
            prefix = sc_embedding[:, :2, :]
            suffix = sc_embedding[:, 3:, :]
            prompt = torch.cat(
                [
                    prefix, 
                    current_style, 
                    suffix
                ], dim=1
            )
        return prompt.cpu(), tokenized_sc.cpu()
    def train_data(self):
        train_data = {
            "classnames": self.classnames, 
            "generator": self, 
            "n_cls": self.n_cls, 
            "n_style": self.n_style, 
        }
        return train_data