from dassl.modeling.backbone import load_clip, clip
from torch import nn
import torch
import torch.nn.functional as F

class StyleGenerator(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.classnames = classnames
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.PROMPTSTYLER.N_STYLE
        self.clip_model = load_clip(cfg)
        self.ctx_dim = self.clip_model.ln_final.weight.shape[0]
        self.init_template()
        ctx_vectors = torch.empty(self.n_style, 1, self.ctx_dim, dtype=torch.float32)
        ctx_vectors.requires_grad_(True)
        ctx_vectors = nn.init.normal_(ctx_vectors, std=0.02)
        self.style_embedding = nn.Parameter(ctx_vectors)
    
    def init_template(self):
        self.template = "a X style of a"
        with torch.no_grad():
            self.s_tokenize = clip.tokenize(self.template).cuda()
            self.s_embedding = self.clip_model.token_embedding(self.s_tokenize)

            self.c_tokenize = clip.tokenize(self.classnames).cuda()
            self.c_output = self.clip_model.encode_text(self.c_tokenize)
            self.c_output = F.normalize(self.c_output, dim=1)

    def forward(self, style_idx):
        current_style = self.style_embedding[style_idx:style_idx+1, :, :]
        if style_idx > 0:
            prefix = self.s_embedding[:, :2, :]
            suffix = self.s_embedding[:, 3:, :]
            prompt = torch.cat(
                [
                    prefix, 
                    current_style, 
                    suffix
                ], dim=1
            )

            output = self.clip_model.forward_text(prompt, self.s_tokenize)
            output = F.normalize(output, dim=1)

            with torch.no_grad():
                before_styles = self.style_embedding[:style_idx, :, :]
                before_prefix = prefix.repeat(style_idx, 1, 1)
                before_suffix = suffix.repeat(style_idx, 1, 1)
                before_prompt = torch.cat(
                    [
                        before_prefix, 
                        before_styles, 
                        before_suffix
                    ], dim=1
                )
                before_output = self.clip_model.forward_text(before_prompt, self.s_tokenize.repeat(style_idx, 1))
                before_output = F.normalize(before_output, dim=1)
            cos_sim = F.cosine_similarity(output.repeat(style_idx, 1), before_output, dim=1)
            style_diversity_loss = torch.abs(cos_sim).mean()
        else:
            style_diversity_loss = torch.tensor(0.0)
        
        sc_template = []
        for cls_idx in range(self.n_cls):
            sc_template.append(self.template + " " + self.classnames[cls_idx])
        with torch.no_grad():
            sc_tokenize = clip.tokenize(sc_template).cuda()
            sc_embedding = self.clip_model.token_embedding(sc_tokenize)
        
        sc_prefix = sc_embedding[:, :2, :]
        sc_suffix = sc_embedding[:, 3:, :]
        sc_prompt = torch.cat(
            [
                sc_prefix, 
                current_style.repeat(self.n_cls, 1, 1), 
                sc_suffix
            ], dim=1
        )
        sc_output = self.clip_model.forward_text(sc_prompt, sc_tokenize)
        sc_output = F.normalize(sc_output, dim=1)
        assert self.c_output.shape == sc_output.shape

        zimn = F.cosine_similarity(sc_output.unsqueeze(1), self.c_output.unsqueeze(0), dim=2)
        exp_zimn = torch.exp(zimn)
        zimn_norm = exp_zimn / exp_zimn.sum(dim=1, keepdim=True)
        content_consistency_loss = -torch.log(zimn_norm.diag()).mean()
        return content_consistency_loss, style_diversity_loss