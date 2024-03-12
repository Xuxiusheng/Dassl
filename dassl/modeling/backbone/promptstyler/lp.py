from dassl.modeling import load_clip, clip
from torch import nn
import torch
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.clip_model = load_clip(cfg)
        self.classifier = nn.Linear(self.clip_model.visual.output_dim, len(classnames), bias=False)
        self.eps = cfg.TRAINER.PROMPTSTYLER.EPS
        self.s = cfg.TRAINER.PROMPTSTYLER.S
        self.m = cfg.TRAINER.PROMPTSTYLER.M

    def arcface_loss(self, input_x, labels):
        for W in self.classifier.parameters():
            W = F.normalize(W, p=2, dim=1)
        wf = self.classifier(input_x)
        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L), wf

    def forward(self, prompts, tokenizes):
        return self.clip_model.forward_text(prompts, tokenizes)

    def forward_img(self, image):
        output = self.clip_model.encode_image(image)
        output = F.normalize(output, dim=1)
        y_class = self.classifier(output)
        return y_class