import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from utils.get_surface import construct_surface, get_kpconv_batch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
import os
import pickle
# from keras.models import Model
from model.electrostatic_encoder import ElectrostaticGridEncoder
from model.surface_encoder import SurfaceFeatureEncoder
from model.extract_electrostatic import *

def graph_poolings(graph_pool,node_representation,batch):
    emb_dim=""

    num_tasks=1  
    #Different kind of graph pooling
    if graph_pool == "sum":
        pool = global_add_pool
    elif graph_pool == "mean":
        pool = global_mean_pool
    elif graph_pool == "max":
        pool = global_max_pool
    elif graph_pool == "attention":
        pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
    elif graph_pool[:-1] == "set2set":
        # set2set_iter = int(graph_pool[-1])
        # pool = Set2Set(emb_dim, set2set_iter)

        pool=Set2Set(
            in_channels=emb_dim, processing_steps=5, num_layers=2)
    else:
        raise ValueError("Invalid graph pooling type.")

    #For graph-level binary classification
    if graph_pool[:-1] == "set2set":
        mult = 2
    else:
        mult = 1

    graph_pred_linear = torch.nn.Linear(mult * emb_dim, num_tasks)

    return graph_pred_linear(pool(node_representation, batch)) 


class Pocket_encoder(torch.nn.Module):
    # def __init__(self,emb_dim="",drop_ratio = 0, graph_pooling = "mean", gnn_type = "gat", encoder_config=""):
    def __init__(self,encoder_config, device):
        super(Pocket_encoder,self).__init__()

        self.device=device
        self.hide_dim = encoder_config['hide_dim']
        self.feat_size = 0
        self.HGNN_train=encoder_config['encoder_Train']['HGNN']
        self.encoder_config_sequence = encoder_config['pocket_sequence']
        self.first_hyperedge=encoder_config['encoder_HGNN']['first_hyperedge']
        self.space_edge=encoder_config['HGNN_first_hyperedge']['space_edge']
        self.knn_edge=encoder_config['HGNN_first_hyperedge']['knn_edge']
        self.pocket_elec = encoder_config['pocket_electrostatic']
        self.pocket_surf = encoder_config['surface_feature']
        self.surf_config = encoder_config['surface_encoder']

        # combined layers
        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(768,256)
        self.fc3=nn.Linear(512,256)


        self.W_matrix=encoder_config['encoder_HGNN']['W_matrix']

        if self.HGNN_train:
            if encoder_config['encoder_HGNN']['first_hyperedge']:
                self.emb_hgnn=HGNN(encoder_config['HGNN_first_hyperedge'])
                self.feat_size = self.feat_size+1
            else:
                raise ValueError("Error")
            
        if self.pocket_elec:
            self.elec_extract = ElectrostaticGridGenerator()    
            self.elec_encoder = ElectrostaticGridEncoder(grid_dims=(129,129,129))  #
            self.feat_size = self.feat_size+1

        if self.pocket_surf:
            self.surf_encoder = SurfaceFeatureEncoder(encoder_config['surface_encoder']).to(self.device)
            self.feat_size = self.feat_size+1

        self.protein_fusion = ProteinFusion(self.hide_dim, self.feat_size, device)
        
        
        
    def forward(self, data, hgnn_data):
        all_feats = []
        if self.HGNN_train:
            if self.first_hyperedge:
                if (self.space_edge and self.knn_edge):
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                elif self.knn_edge:
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                elif self.space_edge:
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                else:
                    raise ValueError("Error")
            else:
                raise ValueError("Error")
            all_feats.append(node_hgnn)

        if self.pocket_surf:
            surface_features = []
            surf_path = './surf/'
            for pocket_file, name in zip(data.pocket_path, data.pocket_name):
                surf_file = os.path.join(surf_path, name.replace('.pdb', '.pkl'))
                surf_data = None
                if  os.path.exists(surf_file):
                    with open(surf_file, 'rb') as f:
                        surf_data = pickle.load(f)
                    xyz = surf_data['xyz']
                    normal = surf_data['normal'] 
                    curvature = surf_data['curvature']   
                else:
                    xyz, normal, curvature, atom, type = construct_surface(pocket_file)

                surf_feat_tensor = torch.cat([
                    torch.from_numpy(xyz),
                    torch.from_numpy(normal),
                    torch.from_numpy(curvature)
                ], dim=1).to(self.device)
                batch = get_kpconv_batch(
                    torch.from_numpy(xyz),  
                    self.surf_config,  
                    surf_feat_tensor  
                )
                batch = move_dict_to_device(batch, self.device)

                surface_feature = self.surf_encoder(surf_feat_tensor, batch)
                surface_feature = surface_feature.mean(dim=0, keepdim=True)

                surface_features.append(surface_feature)
            surf_features = torch.stack(surface_features).squeeze(1)
            all_feats.append(surf_features)
        
        if self.pocket_elec:
            electrostatic_features = []
            grid_path = './elec/'
            for pocket_file, name in zip(data.pocket_path, data.pocket_name):
               
                grid_file = os.path.join(grid_path, name.replace('.pdb', '.pkl'))
                grid_tensor = None
                if os.path.exists(grid_file):
                    with open(grid_file, 'rb') as f:
                        grid_tensor = pickle.load(f)
                        if isinstance(grid_tensor, torch.Tensor):
                            grid_tensor = grid_tensor.to(self.device)
                        else:
                            grid_tensor = None
        
                if grid_tensor is not None:
                    elec_feature = self.elec_encoder(grid_tensor)  # (1, hide_dim)
                    electrostatic_features.append(elec_feature.squeeze(0))
                    del grid_tensor, elec_feature
                else:
                    print(f"Errors when handling pocket {pocket_file}")
                    backup_feature = torch.zeros(self.elec_encoder.fc.out_features, device=self.device)
                    electrostatic_features.append(backup_feature)

            elec_features = torch.stack(electrostatic_features) if electrostatic_features else None
            all_feats.append(elec_features)

        fused_feature= self.protein_fusion(all_feats)


        return fused_feature, all_feats

class HGNN(nn.Module):
    def __init__(self, HGNN_config):
        super(HGNN, self).__init__()
        self.W_matrix=HGNN_config['W_matrix']

        num_features = HGNN_config['num_features']
        n_hid=HGNN_config['hidden_size']
        out_size=HGNN_config['out_size']
        self.dropout=0.3
        use_attention=False
        self.hgc1 = HypergraphConv(in_channels=num_features, out_channels=n_hid, use_attention=use_attention)
        self.hgc2 = HypergraphConv(in_channels=n_hid, out_channels=out_size, use_attention=use_attention)

        self.set2set = Set2Set(in_channels=out_size, processing_steps=5, num_layers=2)

    def forward(self, x, hyperedge_index, batch):

        x = F.relu(self.hgc1(x, hyperedge_index))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, hyperedge_index)
        x = F.elu(x)
        hyperedge_features=None
        return self.set2set(x, batch), x, batch

    
    
class ProteinFusion(nn.Module):
    def __init__(self, hide_dim, feat_size ,device, dropout=0.1):
        super().__init__()

        # Feature Importance Weight
        self.feature_weights = nn.Parameter(torch.ones(feat_size))
        self.device = device

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hide_dim,
            num_heads=4,
            batch_first=True
        ) 
        self.interact_dim = 64

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hide_dim, self.interact_dim),
                nn.ReLU()
            ) for _ in range(feat_size)
        ])

        self.norm1 = nn.LayerNorm(hide_dim)

        self.cross_fusion = nn.Sequential(
            nn.Linear(self.interact_dim * self.interact_dim, hide_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gate_layer = nn.Sequential(
            nn.Linear(hide_dim * 2, hide_dim // 2),
            nn.ReLU(),
            nn.Linear(hide_dim // 2, hide_dim),
            nn.Sigmoid()
        )

        self.norm2 = nn.LayerNorm(hide_dim)
    

    def forward(self, all_feats):
        # [Batch, N_views, Dim] -> [B, 3, 256]
        stacked_feats = torch.stack(all_feats, dim=1) 

        weights = F.softmax(self.feature_weights, dim=0)
        weighted_feats = stacked_feats * weights.view(1, -1, 1)
        attn_out, _ = self.fusion_attention(weighted_feats, weighted_feats, weighted_feats)
        semantic_feat = self.norm1(weighted_feats + attn_out) 
        global_semantic = semantic_feat.mean(dim=1) 
        proj_feats = [proj(feat) for proj, feat in zip(self.projections, all_feats)]
        
        cross_features = []
        num_views = len(all_feats)
        for i in range(num_views):
            for j in range(i + 1, num_views):
                feat_i = proj_feats[i]
                feat_j = proj_feats[j]
                cross_matrix = torch.bmm(feat_i.unsqueeze(2), feat_j.unsqueeze(1))
                cross_flat = cross_matrix.view(cross_matrix.size(0), -1) # [B, 64*64]
                cross_out = self.cross_fusion(cross_flat) # [B, 256]
                cross_features.append(cross_out)
        if cross_features:
            interaction_feat = torch.stack(cross_features, dim=1).mean(dim=1) # [B, 256]
        else:
            interaction_feat = torch.zeros_like(global_semantic)
        combined = torch.cat([global_semantic, interaction_feat], dim=1)
        z = self.gate_layer(combined)
        
        fused_feat = z * global_semantic + (1 - z) * interaction_feat

        fused_feat = self.norm2(fused_feat)
        

        return fused_feat

def move_dict_to_device(data_dict, device):
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device)
        elif isinstance(value, dict):
            data_dict[key] = move_dict_to_device(value, device)
    return data_dict
