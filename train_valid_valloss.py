import sys
sys.path.append('')
import yaml
import os
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from model.decoder import GPTConfig
import time
from rdkit import Chem
from generate.sample_util import sample
from dataloaders.dataloader_protein_atom import datapair_loader,initial_vocab
from model.model import HoS_Gen
import warnings
from generate.sample_util  import convert_smiles, inverted_dict
warnings.filterwarnings("ignore")
import os
import metrics.SA_Score.sascorer as sascorer
import numpy as np
from rdkit.Chem import QED
from transformers import get_linear_schedule_with_warmup

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

if __name__ == "__main__":
    
    # load configuration file
    train_dir="./config/train.yaml"
    encoder_dir="./config/encoder.yaml"
    decoder_dir="./config/decoder.yaml"
    with open(train_dir, 'r') as f:
        train_config = yaml.full_load(f)

    with open(encoder_dir, 'r') as f:
        encoder_config = yaml.full_load(f)
    
    with open(decoder_dir, 'r') as f:
        decoder_config = yaml.full_load(f)
    hgnn_train=encoder_config['encoder_Train']['HGNN']

    if hgnn_train:
        flag='all'
    else:
        flag=''

    if decoder_config['pretrain']:
        train_config['out_dir'] = train_config['out_dir'] + 'pre/'
        out_dir = os.path.join(train_config['out_dir'],train_config['dataset'],train_config['vocab'],flag) + '/'
    else:
        out_dir = os.path.join(train_config['out_dir'],train_config['dataset'],train_config['vocab'],flag) + '/'

    local=time.strftime('%Y_%m_%d_%H') # local=time.strftime('%Y_%m_%d_%H_%M_%S') 
    out_dir=out_dir+local+'/'
    train_config['out_dir']=out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('results saved in {}'.format(out_dir))
    trained_model_dir = out_dir

    run_name=local
    # detect cpu or gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
                      
    # random.seed(train_config['seed'])
    set_seed(train_config['seed'])  

    # training data files
    pocket_dir = train_config['pocket_dir']
    features_to_use = encoder_config['features_to_use']
    dataset_type=train_config['dataset']

    # load the pocket-smiles pairs 
    smiles_train_dict, smiles_val_dict=data_split(train_config['smiles_dir'], dataset_type)

    # dataloaders
    batch_size = train_config['batch_size']
    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    # num_workers = 10
    print('number of workers to load data: ', num_workers)


    trainloader, train_size, train_max_len = datapair_loader(
        smiles_train_dict,
        pocket_dir,
        features_to_use,
        dataset_type,
        vocab=train_config['vocab'],
        vocab_path=train_config['vocab_path'],
        batch_size=batch_size, shuffle=True,
        hgnn_train=hgnn_train,
        num_workers=num_workers
    )
    print('size of train set: ', train_size)

    valloader, val_size, val_max_len = datapair_loader(
        smiles_val_dict,
        pocket_dir,
        features_to_use,
        dataset_type,
        vocab=train_config['vocab'],
        vocab_path=train_config['vocab_path'],
        batch_size=batch_size, shuffle=False,
        hgnn_train=hgnn_train,
        num_workers=num_workers
    )
    print('size of val set: ', val_size)

    val_genloader, val_gen_size, gen_max_len = datapair_loader(
        smiles_val_dict,
        pocket_dir,
        features_to_use,
        dataset_type,
        vocab=train_config['vocab'],
        vocab_path=train_config['vocab_path'],
        batch_size=1, shuffle=False,
        hgnn_train=hgnn_train,
        num_workers=1
    )

    
    max_len=max(train_max_len,val_max_len)
    decoder_config['block_size']=max_len
    print("max_len",max_len)
    
    # model initialization
    config=GPTConfig(decoder_config['vocab_size'], max_len, num_props=int(decoder_config['num_props']),n_layer=decoder_config['n_layer'],
                     n_embd=decoder_config['n_embd'],n_head=decoder_config['n_head'],pretain=decoder_config['pretrain'],att_num=int(decoder_config['att_num']),
                     alpha_use=decoder_config['use_alpha'],use_gate=decoder_config['use_gate'],use_encoder_norm=decoder_config['use_encoder_norm'],
                     use_alpha=decoder_config['use_alpha'])
    
    model = HoS_Gen(
        train_config,
        encoder_config,
        config,
        device
        ).to(device)

    # save config      # save val_set
    save_config(out_dir,train_config,"train")
    save_config(out_dir,encoder_config,"encoder")
    save_config(out_dir,decoder_config,"decoder")
    save_config(out_dir,smiles_val_dict,"val_dict")

    # load pretrained decoder
    if decoder_config['pretrain']:
        print('loading pretrained decoder...')
        model.decoder.load_state_dict(
            torch.load(
                decoder_config['pretrained_model'],
                map_location=torch.device(device)
            ),
            strict=False
        )
        print('Pretrained for decoder is loaded.')
    else:
        print('No pretraining for decoder.')

    # the optimizer
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']

    optimizers_config = TrainerConfig(
        train_config
    )
    optimizer = configure_optimizers(model, optimizers_config)

    num_epoch = train_config['num_epoch']
    # the learning rate scheduler
    num_training_steps = len(trainloader) * num_epoch
    warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # get the index of padding # PADDING_IDX = decoder_config - 1
    PADDING_IDX=decoder_config['vocab_size']-1
    loss_function = nn.CrossEntropyLoss(ignore_index=PADDING_IDX, reduction='sum')
    # loss_function = nn.CrossEntropyLoss(reduction='sum')

    # train and validation, the results are saved.
    # cl_losses = []
    train_losses = []
    val_losses = []
    qed_all = []
    sa_all = []
    valid_all=[]
    best_qed = 0
    bset_sa = 0
    best_val_loss, best_val_epoch = float('inf'), None   # best_val_loss, best_val_epoch = float('inf'), None
    best_qed_loss, best_qed_epoch = 0, None
    num_epoch = train_config['num_epoch']
    print('begin training...')
    best_epoch= None
    for epoch in range(1, 1 + num_epoch):
        # train
        model.train()
        c_loss = 0 # add
        train_loss = 0
        for data, hgnn_data in trainloader:
            optimizer.zero_grad()
            data = data.to(device, non_blocking=True)
            hgnn_data=hgnn_data.to(device, non_blocking=True)
            smiles = data.y
            input=data.input
            targets=data.target 

            # the lengths are decreased by 1 because we don't use <eos> for input and we don't need <sos> for
            # output during traning.
            lengths = [len(x) - 1 for x in smiles]
            smiles = [torch.tensor(x) for x in smiles]
            input_smiles = [x[:-1] for x in smiles]
            targets=[x[1:] for x in smiles]
        
            smiles = pad_sequence(
                input_smiles, batch_first=True,
                padding_value=PADDING_IDX
            ).to(device)
            # forward  
            preds, _= model(
                data, 
                hgnn_data, 
                smiles, 
                lengths
                )  

            targets=pad_sequence(                       # [batch_size, max_sequence_length]
                targets, batch_first=True,
                padding_value=PADDING_IDX
            ).to(device) 

            loss = loss_function(preds, targets.view(-1))
            # loss.backward(retain_graph=True)  
            # loss.backward()
            # total_loss = loss + cl_loss
            total_loss = loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss = loss.detach()
            # cl_loss = cl_loss.detach()
            train_loss = train_loss + total_loss.item()  # * data.num_graphs
            # add
            # c_loss = c_loss + cl_loss.item()

            torch.cuda.empty_cache()
        
        train_losses.append(train_loss / train_size)
        # cl_losses.append(c_loss / train_size)

        # validation
        model.eval()
        val_loss = 0
        for data, hgnn_data in valloader:
            data = data.to(device)
            hgnn_data=hgnn_data.to(device)
            smiles = data.y
            input=data.input
            targets=data.target 

            lengths = [len(x) - 1 for x in smiles]
            smiles = [torch.tensor(x) for x in smiles]
            input_smiles = [x[:-1] for x in smiles]
            targets=[x[1:] for x in smiles]
        
            smiles = pad_sequence(
                input_smiles, batch_first=True,
                padding_value=PADDING_IDX
            ).to(device)



            with torch.no_grad():
                preds, loss= model(
                    data, 
                    hgnn_data, 
                    smiles, 
                    lengths
                    )  
            targets=pad_sequence(
                targets, batch_first=True,
                padding_value=PADDING_IDX
            ).to(device) 

            loss = loss_function(preds, targets.view(-1))
            val_loss += loss.item()

        val_losses.append(val_loss / val_size)


        pocket_num = 0
        molecules_num = 20   
        temperature = 1.0
        valset_vocab = initial_vocab(train_config['vocab'],train_config['vocab_path'])
        top_k=30
        qed_each_pokcet = []
        sa_each_pokcet = []
        valid_pocket=[]
        inverted_vocab=inverted_dict(train_config['vocab_path'])
        for i,(data, hgnn_data) in enumerate(val_genloader):
            if pocket_num>50:
                break
            if pocket_num<100:
                data = data.to(device)
                hgnn_data=hgnn_data.to(device)
                pocket_name = data.pocket_name[0]
                pocket_name = os.path.splitext(os.path.basename(pocket_name))[0]
                start_int = [key for key, value in valset_vocab.int2tocken.items() if value == '<sos>'][0]
                # create a tensor of shape [batch_size, seq_step=1]
                sos = torch.ones(
                    [molecules_num, 1],
                    dtype=torch.long,
                    device=device
                )
                sos = sos * start_int
                x = torch.tensor(start_int, dtype=torch.long,device=device)[None,...].repeat(molecules_num, 1)

                molecules= sample(model, data, hgnn_data, x, decoder_config['block_size'], device, valset_vocab, molecules_num, temperature=1, sample=True, top_k=top_k)

                num_invalid=0
                num_valid=0
                qed_each_smiles=[]
                sa_each_smiles=[]
                for smiles in molecules:
                    smiles=convert_smiles(train_config['vocab'], inverted_vocab, smiles.tolist())
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print('SMILES of None value in sample',smiles)
                        num_invalid += 1
                        continue
                    else:
                        num_valid += 1
                        Chem.SanitizeMol(mol)
                        try:
                            qed_every = QED.qed(mol)
                        except Exception as e:
                            qed_every = 0
                        qed_each_smiles.append(float(qed_every))
                        try:
                            # sa_every = round((10-sascorer.calculateScore(mol))/9)
                            sa_every = round((10-sascorer.calculateScore(mol))/9,2)
                        except Exception as e:
                            sa_every = 0
                        sa_each_smiles.append(float(sa_every))
                if len(qed_each_smiles) == 0:
                    qed_mean=0
                else:
                    qed_mean =  sum(qed_each_smiles) /len(qed_each_smiles)
                    qed_each_pokcet.append(float(qed_mean))

                if len(sa_each_smiles) == 0:
                    sa_mean=0
                else:
                    sa_mean =  sum(sa_each_smiles) /len(sa_each_smiles)
                    sa_each_pokcet.append(float(sa_mean))

                valid_pocket.append(float(num_valid/(num_valid+num_invalid)))

            pocket_num = pocket_num + 1

        if len(qed_each_pokcet) ==0:
            qed_all.append(0)
        else:
            qed_all.append(sum(qed_each_pokcet) /len(qed_each_pokcet))
        
        if len(sa_each_pokcet) ==0:
            sa_all.append(0)
        else:
            sa_all.append(sum(sa_each_pokcet) /len(sa_each_pokcet))

        valid_all.append(sum(valid_pocket)/len(valid_pocket))

        print('epoch {}, train loss: {}, val loss: {}'.format(
            epoch, train_losses[-1], val_losses[-1]))

        # update the saved model upon best validation loss
        if qed_all[-1] >= best_qed:
            best_qed_epoch = epoch
            best_qed = qed_all[-1]
            torch.save(model.state_dict(), trained_model_dir + 'qed_model.pt')
            print('model saved at epoch {}'.format(epoch))

        if val_losses[-1] <= best_val_loss:
            best_val_epoch = epoch
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), trained_model_dir + 'val_model.pt')
            print('model saved at epoch {}'.format(epoch))

        if epoch % 5 == 0 or epoch == 1:
            model_filename = trained_model_dir + f'{epoch}.pt'
            torch.save(model.state_dict(), model_filename)
            print('model saved at epoch {}'.format(epoch))

        # scheduler.step(val_losses[-1])
        # scheduler.step(qed_all[-1]) 
        # wandb.log({'step_train_loss': val_losses, 'train_step': epoch + epoch*len(trainloader), 'learning_rate': optimizer.state_dict()['param_groups'][0]['lr']})

        loss_history = [train_losses, val_losses, qed_all, sa_all, valid_all]
        print("best_val_epoch", best_val_epoch)
        with open(out_dir + 'loss.yaml', 'w') as f:
            yaml.dump(loss_history, f)

    # save train and validation losses
    loss_history = [train_losses, val_losses, qed_all, sa_all, valid_all]
    print("best_val_epoch", best_val_epoch)
    print("best_qed_epoch", best_qed_epoch)
    with open(out_dir + 'loss.yaml', 'w') as f:
        yaml.dump(loss_history, f)
    # wandb.finish()







