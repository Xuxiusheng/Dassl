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
        self.n_cls = len(classnames)
        ctx_vectors = torch.empty(self.n_styles, 1, self.ctx_dim, dtype=torch.float32)
        ctx_vectors.requires_grad_(True)
        ctx_vectors = nn.init.normal_(ctx_vectors, std=0.02)
        self.style_embedding = nn.Parameter(ctx_vectors)

        self.text = "a X style of a"

        with torch.no_grad():
            tokenize_c = clip.tokenize(self.classnames).cuda()
            c_output = self.clip_model.encode_text(tokenize_c)
            self.c_norm = F.normalize(c_output, dim=1)

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

        content_consistency_loss = torch.tensor(0.0)
        for s in range(self.n_styles):
            current_style = self.style_embedding[s:s+1, :, :]
            sc_template = []
            for cls_idx in range(self.n_cls):
                sc_template.append(self.text + " " + self.classnames[cls_idx])
            with torch.no_grad():
                tokenized_sc = clip.tokenize(sc_template).cuda()
                sc_embedding = self.clip_model.token_embedding(tokenized_sc)
                sc_prefix = sc_embedding[:, :2, :]
                sc_suffix = sc_embedding[:, 3:, :]
            sc_prompt = torch.cat(
                [
                    sc_prefix, 
                    current_style.repeat(self.n_cls, 1, 1), 
                    sc_suffix
                ], dim=1
            )
            sc_outputs = self.clip_model.forward_text(sc_prompt, tokenized_sc)
        
            sc_norm = F.normalize(sc_outputs, dim=1)

            zimm = F.cosine_similarity(sc_norm.unsqueeze(1), self.c_norm.unsqueeze(0), dim=2)
            exp_zimm = torch.exp(zimm)
            per_zimm = exp_zimm / exp_zimm.sum(dim=1, keepdim=True)

            content_consistency_loss += -torch.log(per_zimm.diag()).mean()
        
        return diversity_loss, content_consistency_loss / self.n_styles

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