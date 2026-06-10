from generate.sample_util import check_novelty, sample_RL, canonic_smiles,inverted_dict,convert_smiles,obey_lipinski,calculate_diversity,set_seed
from dataloaders.dataloader_protein_atom import datapair_loader,initial_vocab
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import RDConfig
import yaml
import os
import sys
sys.path.append('..')
from model.model import Pocket_GNN
from model.decoder import GPTDecoder, GPTConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from collections import Counter
import time
import metrics.SA_Score.sascorer as sascorer
import metrics.NP_Score.npscorer as npscorer
from dataloaders.dataloader_protein_atom import DatapairDataset
from torch_geometric.loader import DataLoader
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from dataloaders.dataloader_protein_atom import pocket_sequence_gen
import warnings
import random
from model.reward_score import RewardScore
import gc
warnings.filterwarnings("ignore")

def read_config(out_dir, config_name):
    config_dir = out_dir + config_name+".yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)
    return config

def data_dict(smiles_dir,val_dir,test_dir):
    with open(val_dir + 'val_dict.yaml', "r") as file_b:
        data_val = yaml.safe_load(file_b)
        dict_val = data_val
    with open(smiles_dir, "r") as file_a:
        data_a = yaml.safe_load(file_a)
        dict_train = {k: v for k, v in data_a.items() if k not in dict_val}


    return dict_val, dict_train

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def data_split(data_path, dataset):
    with open(data_path, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)
    keys = list(data.keys())
    random.shuffle(keys)

    total_data = len(keys)
    num_parts = 10

    base_count = total_data // num_parts
    remainder = total_data % num_parts

    val_size = base_count + 1 if remainder != 0 else base_count
    val_set = {key: data[key] for key in keys[:val_size]}
    train_set = {key: data[key] for key in keys[val_size:]}

    return train_set, val_set


def full_clean():
    for var in ['molecules', 'log_probs', 'new_log_probs', 'old_log_probs', 'new_log_probs_cut',
                'ratio', 'expanded_rewards', 'surr1', 'surr2', 'probs', 'total_loss', 'ppo_loss', 'entropy_loss',
                'rewards_tensor', 'normalized_rewards', 'qeds', 'sass', 'rewards']:
        if var in locals():
            del locals()[var]
    
    torch.cuda.empty_cache() 

    import gc
    gc.collect()
    gc.collect()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

def save_config(out_dir, config ,config_name):
    with open(out_dir + config_name +'.yaml', 'w') as f:
        yaml.dump(config, f)

class TrainerConfig:
    def __init__(self, train_config, **kwargs):
        self.max_epochs = train_config['num_epoch']  
        self.batch_size = train_config['batch_size'] 
        self.latent_dim = train_config['hide_dim']
        self.learning_rate = train_config['learning_rate']  
        self.betas = (0.9, 0.95)  
        self.weight_decay = train_config['weight_decay'] 
        self.grad_norm_clip = 0.8 
        self.lr_decay = True  
        self.warmup_tokens = 375e6  
        self.final_tokens = 260e9  
        self.ckpt_path = None  
        self.num_workers = 0 
        for k, v in kwargs.items():
            setattr(self, k, v)

def configure_optimizers(self, train_config): 
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    param_dict = {pn: p for pn, p in self.named_parameters()}
    for mn, m in self.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if "bias" in pn:
                no_decay.add(fpn)
            elif "weight" in pn and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif "ln" in pn or "norm" in pn:
                no_decay.add(fpn)
            elif "emb" in pn:
                no_decay.add(fpn)
            else:
                decay.add(fpn)
    if "pos_emb" in param_dict:
        no_decay.add("pos_emb")
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"The parameter {inter_params} exists in both the decay and no_decay sets!"
    unclassified = param_dict.keys() - union_params
    if unclassified:
        no_decay.update(unclassified)  

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

    for i, group in enumerate(optim_groups):
        print(f"Optimizer group {i}:")
        print(f"  Parameters: {len(group['params'])}")
        print(f"  Weight decay: {group['weight_decay']}")

    return optimizer


def safe_apply(func, molecule):
    try:
        return func(molecule)
    except Exception:
        return np.nan

def train(model,training_history, epoch, best_reward, hgnn_data, data, vocab, device):
    start_int = [key for key, value in valset_vocab.int2tocken.items() if value == '<sos>'][0]

    # create a tensor of shape [batch_size, seq_step=1]
    sos = torch.ones(
        [molecule_nums, 1],
        dtype=torch.long,
        device=device
    )
    sos = sos * start_int
    x = torch.tensor(start_int, dtype=torch.long, device=device)[None,...].repeat(molecule_nums, 1)

    molecules, log_probs, entropies= sample_RL(model, data, hgnn_data, x, block_size, device, valset_vocab.vocab, molecule_nums, temperature=temperature, sample=sample_type, top_k=top_k)
    rewards = []
    qeds = []
    sa = []
    valid_count = 0
    num_invalid=0
    mol_list = []

    for smiles in molecules:
        smiles=convert_smiles(vocab, inverted_vocab, smiles.tolist())
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print('SMILES of None value in sample',smiles)
            num_invalid += 1
            rewards.append(0.0)
            qeds.append(0.0)
            sa.append(0.0)
            continue
        else:
            try:
                qed = QED.qed(mol)
            except Exception as e:
                num_invalid += 1
                rewards.append(0.0)
                qeds.append(0.0)
                sa.append(0.0)
                continue
            sas = sascorer.calculateScore(mol)
            sas_0_1 = round((10 - sas) / 9, 2)
            lipinski = obey_lipinski(mol)
            reward = reward_score.calculate_score(qed, sas_0_1, mol, smiles, data['pocket_path'][0])

            rewards.append(reward)
            qeds.append(qed)
            sa.append(sas_0_1)
            valid_count += 1

            mol_list.append(mol)

    diversity = calculate_diversity(mol_list)
    reward_score.reset_batch_memory()
    validity = valid_count / train_rl_config['molecule_nums']
    aver_reward = np.mean(rewards) if rewards else 0.0
    aver_qed = np.mean([q for q in qeds if q > 0])
    aver_sas = np.mean([s for s in sa if s > 0]) 


    print(f"Processing pocket: {data.pocket_name}")
    print(f"  Validity: {validity:.2%}")
    print(f"  Avg Reward: {aver_reward:.4f}")
    print(f"  Avg QED: {aver_qed:.4f}")
    print(f"  Avg SA: {aver_sas:.4f}")
    print(f" Diversity: {diversity:.4f} ")

    if valid_count > 0:
        model.train()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        normalized_rewards = rewards_tensor - rewards_tensor.mean()
        new_logits = model.sample_from_pocket(data, hgnn_data, molecules, lengths="")
        new_logits = new_logits[:, :-1, :]      
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        new_log_probs = new_log_probs.gather(2, molecules[:, 1:].unsqueeze(2)).squeeze(2)

        min_len = min(log_probs.size(1), new_log_probs.size(1))
        old_log_probs = log_probs[:, :min_len]
        new_log_probs = new_log_probs[:, :min_len]

        ratio = torch.exp(new_log_probs - old_log_probs)

        expanded_rewards = normalized_rewards.unsqueeze(1).expand_as(ratio)
 
        surr1 = ratio * expanded_rewards
        surr2 = torch.clamp(ratio, 1 - train_rl_config['ppo_epsilon'], 1 + train_rl_config['ppo_epsilon']) * expanded_rewards
        ppo_loss = -torch.min(surr1, surr2).mean()

        probs = torch.exp(new_log_probs)
        entropy_loss = -torch.mean(torch.sum(probs * new_log_probs, dim=-1))
        
        total_loss = ppo_loss + train_rl_config['entropy_weight'] * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss = total_loss.item()
        print(f"  Loss: {epoch_loss:.4f}")
        full_clean()

    
    else:
        epoch_loss = 0.0
        print("  No valid molecules, skipping update")

    training_history['pocket_name'].append(data.pocket_name)
    training_history['epoch'].append(epoch)
    training_history['avg_reward'].append(float(aver_reward))
    training_history['avg_qed'].append(float(aver_qed))
    training_history['avg_sa'].append(float(aver_sas))
    training_history['validity'].append(float(validity))
    training_history['loss'].append(float(epoch_loss))

    return training_history, best_reward    






if __name__ == '__main__':
    train_rl_config_dir="./config/train_rl.yaml"
    with open(train_rl_config_dir, 'r') as f:
        train_rl_config = yaml.full_load(f)

    top_k=train_rl_config['top_k']
    result_dir=train_rl_config['result_dir']
    molecule_nums=train_rl_config['molecule_nums'] 
    temperature=train_rl_config['temperature']
    train_rl_num=train_rl_config['sample_num']
    sample_type= train_rl_config['sample_type']
    num_epochs = train_rl_config['num_epochs']

    # load the configuartion file in output
    train_config=read_config(result_dir,"train")
    encoder_config=read_config(result_dir,"encoder")
    decoder_config=read_config(result_dir,"decoder")
    block_size=decoder_config['block_size']
    out_dir = train_rl_config['out_dir']
    local=time.strftime('%Y_%m_%d_%H') 
    out_dir=out_dir+local+'/'
    checkpoint_dir = out_dir + 'checkpoint/' 
    train_rl_config['out_dir']=out_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    set_seed(train_config['seed'])

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    print('device: ', device)

    vocab, vocab_path =  train_config['vocab'], train_config['vocab_path']
    inverted_vocab = inverted_dict(train_config['vocab_path'])
    valset_vocab = initial_vocab(train_config['vocab'],train_config['vocab_path'])

    model_path = result_dir + "1.pt"

    config=GPTConfig(decoder_config['vocab_size'], block_size, num_props=int(decoder_config['num_props']),n_layer=decoder_config['n_layer'],
                     n_embd=decoder_config['n_embd'],n_head=decoder_config['n_head'],att_num=int(decoder_config['att_num']),
                     alpha_use=decoder_config['use_alpha'],use_gate=decoder_config['use_gate'],use_encoder_norm=decoder_config['use_encoder_norm'],
                     sample=True,pretain=decoder_config['pretrain'],use_alpha=decoder_config['use_alpha'])
    model = Pocket_GNN(
        train_config,
        encoder_config,
        config,
        device,
        ).to(device)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device)
        ),
        strict=False
    )
    reward_score = RewardScore(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_rl_config['learning_rate']
    )
    training_history = {
        'epoch': [],
        'avg_reward': [],
        'avg_qed': [],
        'avg_sa': [],
        'validity': [],
        'loss': []
    }

    smiles_train_dict, smiles_val_dict = data_split(
        train_config['smiles_dir'],
        train_config['dataset']
    )
    train_smiles_set = set(smiles_train_dict.values())

    pocket_dir = train_config['pocket_dir']
    features_to_use = encoder_config['features_to_use']
    dataset_type=train_config['dataset']

    trainloader, train_size, train_max_len = datapair_loader(
        smiles_train_dict,
        pocket_dir,
        features_to_use,
        dataset_type,
        vocab=train_config['vocab'],
        vocab_path=train_config['vocab_path'],
        batch_size=1,
        shuffle=True,
        hgnn_train=encoder_config['encoder_Train']['HGNN'],
        num_workers=os.cpu_count() // 2
    )
    print('size of train set: ', train_size)

    valloader, val_size, val_max_len = datapair_loader(
        smiles_val_dict,
        pocket_dir,
        features_to_use,
        dataset_type,
        vocab=train_config['vocab'],
        vocab_path=train_config['vocab_path'],
        batch_size=1, shuffle=False,
        hgnn_train=encoder_config['encoder_Train']['HGNN'],
        num_workers=1
    )
    print('size of val set: ', val_size)

    max_len=max(train_max_len,val_max_len)
    decoder_config['block_size']=max_len
    print("max_len",max_len)

    save_config(out_dir, train_config, "train")
    save_config(out_dir, encoder_config, "encoder")
    save_config(out_dir, decoder_config, "decoder")
    save_config(out_dir, train_rl_config, "train_rl")
    save_config(out_dir, smiles_val_dict, "val_dict")
    training_history = {
        'pocket_name': [],
        'epoch': [],
        'avg_reward': [],
        'avg_qed': [],
        'avg_sa': [],
        'validity': [],
        'loss': []
    }

    print('Starting RL training...')
    best_reward = -float('inf')
    train_iter = iter(trainloader)

    for epoch in range(1, num_epochs+1):


        model.train()
        epoch_rewards = []
        epoch_qeds = []
        epoch_sas = []
        epoch_diversities = []
        epoch_validities = []
        epoch_losses = []
        data, hgnn_data = next(train_iter)

        data = data.to(device, non_blocking=True)
        hgnn_data=hgnn_data.to(device, non_blocking=True)

        training_history, best_reward = train(model, training_history, epoch, best_reward, hgnn_data, data, valset_vocab, device)

        if epoch % 100 == 0:
            model_filename = checkpoint_dir + f'{epoch}.pt'
            torch.save(model.state_dict(), model_filename)
            print('model saved at epoch {}'.format(epoch))

        history_path = os.path.join(out_dir, 'rl_training_history.yaml')
        with open(history_path, 'w') as f:
            yaml.dump(training_history, f)

        del hgnn_data, data

    print(f"RL training completed!")
    print(f"Best reward achieved: {best_reward:.4f}")
    print(f"{'='*60}")

