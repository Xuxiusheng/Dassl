from torch import nn
from dassl.modeling.backbone import load_clip, clip
import torch
import numpy as np
import torch.nn.functional as F
import math

class ETFHead(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.clip_model = load_clip(cfg)
        self.n_styles = cfg.TRAINER.BATSTYLER.N_STYLE
        self.generate_orthogonal(self.clip_model.visual.output_dim)
        self.ctx_dim = self.clip_model.ln_final.weight.shape[0]
        self.classnames = classnames
        ctx_vectors = torch.empty(self.n_styles, 1, self.ctx_dim, dtype=torch.float32)
        ctx_vectors.requires_grad_(True)
        ctx_vectors = nn.init.normal_(ctx_vectors, std=0.02)
        self.style_embedding = nn.Parameter(ctx_vectors)

        self.text = "a X style of a"

    def generate_orthogonal(self, in_features):
        rand_mat = np.random.random(size=(in_features, self.n_styles))
        orth_vec, _ = np.linalg.qr(rand_mat)
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(self.n_styles), atol=1.e-7), \
            "The max irregular value is : {}".format(torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(self.n_styles))))
        
        i_nc_nc = torch.eye(self.n_styles)
        one_nc_nc = torch.mul(torch.ones(self.n_styles, self.n_styles), (1 / self.n_styles))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc), math.sqrt(self.n_styles / (self.n_styles - 1)))
        self.etf_vec = etf_vec.cuda()

    def loss(self, x):
        output = torch.exp(torch.mm(x, self.etf_vec))
        label_col = torch.diag(output)
        diversity_loss = -torch.log(label_col / output.sum(dim=1)).mean()
        return diversity_loss

    def forward(self):
        with torch.no_grad():
            tokenize = clip.tokenize(self.text).cuda()
            embedding = self.clip_model.token_embedding(tokenize)
            prefix = embedding[:, :2, :]
            suffix = embedding[:, 3:, :]

        prompt = torch.cat(
            [
                prefix.repeat(self.n_styles, 1, 1), 
                self.style_embedding, 
                suffix.repeat(self.n_styles, 1, 1)
            ], dim=1
        )
        output = self.clip_model.forward_text(prompt, tokenize.repeat(self.n_styles, 1))
        output = F.normalize(output, dim=1)

        return output