import torch
import torch.nn as nn
from model.surface_blocks import block_decider


class SurfaceFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = 0
        self.radius = config['first_subsampling_dl'] * config['conv_radius']
        self.in_dim = config['in_feats_dim']  
        self.out_dim = config['first_feats_dim']
        self.hide_dim = config['hide_dim']

        self.encoder_blocks = nn.ModuleList()
        for block in config['architectures']:
            if 'upsample' in block: 
                break
            self.encoder_blocks.append(block_decider(
                block, self.radius, self.in_dim, self.out_dim, self.layer, config
            ))
            self.in_dim = self.out_dim // 2 if 'simple' in block else self.out_dim
            if 'pool' in block or 'strided' in block:
                self.layer += 1
                self.radius *= 2
                self.out_dim *= 2
        
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim , self.hide_dim),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(self.hide_dim)
        )

    def forward(self, surf_features, batch):
        x = surf_features
        for block_op in self.encoder_blocks:
            x = block_op(x, batch)
        x = self.mlp(x)
        return x 