from torch import nn
from dassl.modeling.backbone import load_clip, clip
import torch.nn.functional as F
import torch

class Decoupler(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = load_clip(cfg)
        self.s = cfg.TRAINER.OUR.S
        self.m = cfg.TRAINER.OUR.M
        self.eps = cfg.TRAINER.OUR.EPS
        self.input_dim = self.clip_model.visual.output_dim
        self.fc = nn.Linear(self.input_dim, len(classnames), bias=False)
        nn.init.xavier_normal_(self.fc.weight)

    def arcface(self, input, labels):
        for param in self.fc.parameters():
            param.data = F.normalize(param.data, dim=1)

        x = F.normalize(input, p=2, dim=1)
        wf = self.fc(x)
        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

    def forward(self, input):
        return self.fc(input)

    def encode_image(self, image):
        image_features = self.clip_model.encode_image(image)
        return image_features