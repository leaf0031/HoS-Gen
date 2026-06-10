import sys
sys.path.append('')
import random
import torch
import os
from torch_geometric.data import Data, Dataset
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
# from tape import ProteinBertModel, TAPETokenizer
from .pocket_utils import pocket_hypergraph
import yaml
from .get_protein_sequence import *
# self-defined utilities
import utils.util as utils

def pocket_sequence_gen(dataset_type):
    data_dir='./data_crossdocked/sequence/pocket_sequence.yaml'
    with open(data_dir, 'r') as f:
        sequences = yaml.full_load(f)
    return sequences


def datapair_loader(smiles_dict,
                    pocket_dir,
                    features_to_use,
                    dataset_type,
                    vocab,
                    vocab_path,
                    batch_size,
                    shuffle=False,
                    hgnn_train=False,
                    num_workers=4,
                    ):
    
    # split pockets into train/test split
    pockets = list(smiles_dict.keys())
    random.shuffle(pockets)

    dataset= DatapairDataset(
        dataset_type,
        # pocket_sequence,
        pockets=pockets,
        pocket_dir=pocket_dir,
        smiles_dict=smiles_dict,
        features_to_use=features_to_use,
        vocab=vocab,
        vocab_path=vocab_path,
        hgnn_train=hgnn_train,
        
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    # max_len = max(
    # (max(data.vocab_len, hgnn_data.vocab_len) for data, hgnn_data in dataset),
    # key=lambda vocab_len: vocab_len)
    max_len = 92

    # max_len = max(dataset, key=lambda data: data.vocab_len).vocab_len
    # max_len=max_len = max(dataset[0], key=lambda data: data.vocab_len).vocab_len

    return dataloader, len(dataset), max_len


class DatapairDataset(Dataset):
    def __init__(self,
                 dataset_type,
                 pockets,
                 pocket_dir,
                 smiles_dict,
                 features_to_use,
                 vocab,
                 vocab_path,
                 hgnn_train,
                 ):
        
        self.dataset_type=dataset_type
        # self.pocket_sequence=pocket_sequence
        self.pockets = pockets
        self.pocket_dir = pocket_dir
        self.smiles_dict = smiles_dict
        self.hgnn_train=hgnn_train

        # distance threshold to form an undirected edge between two atoms
        self.threshold = 4.5

        # hard coded info to generate 2 node features
        self.hydrophobicity = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
                               'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
                               'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
                               'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
                               'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
        self.binding_probability = {'ALA': 0.701, 'ARG': 0.916, 'ASN': 0.811, 'ASP': 1.015,
                                    'CYS': 1.650, 'GLN': 0.669, 'GLU': 0.956, 'GLY': 0.788,
                                    'HIS': 2.286, 'ILE': 1.006, 'LEU': 1.045, 'LYS': 0.468,
                                    'MET': 1.894, 'PHE': 1.952, 'PRO': 0.212, 'SER': 0.883,
                                    'THR': 0.730, 'TRP': 3.084, 'TYR': 1.672, 'VAL': 0.884}
        
        total_features = ['x', 'y', 'z', 'r', 'theta', 'phi',
                          'hydrophobicity', 'binding_probability','atom']

        # features to use should be subset of total_features
        assert(set(features_to_use).issubset(set(total_features)))
        self.features_to_use = features_to_use

        # initialize the vocabulary used to tokenize smiles
        if vocab == 'char':
            self.vocab = utils.CharVocab(vocab_path)
        elif vocab == 'selfies':
            self.vocab = utils.SELFIESVocab(vocab_path)
        elif vocab == 'regex':
            self.vocab = utils.RegExVocab(vocab_path)
        elif vocab=="moses":
            pass
        else:
            raise ValueError("invalid vocab value.")


    def __len__(self):
        return len(self.pockets)

    def len(self):
        return len(self.pockets)

    def get(self):
        pass
    
    def __getitem__(self, idx):
        pocket = self.pockets[idx]

        if self.dataset_type=='case_study':
            pocket_dir = self.pocket_dir

        if self.dataset_type=='crossdocked':
            pocket_dir = self.pocket_dir + pocket

        if self.dataset_type=='pdbbind':
            pocket_dir = self.pocket_dir + pocket + '/' + pocket + '_pocket.pdb'

        hgnn_x, edge_index_space, edge_index_sequence, edge_index_first = None, None, None, None
        hgnn_data=None
        data = None

        if self.hgnn_train:

            hgnn_x, edge_index_space, edge_index_sequence, edge_index_first = pocket_hypergraph(
                pocket_dir, self.threshold
            )
            edge_attr = None

            hgnn_data = Data(x=hgnn_x,edge_index=edge_index_space,edge_attr=edge_attr) 

        if hgnn_x is None :
            data=Data()
            hgnn_data=Data()
        else:
            data=hgnn_data
        

        vocab_len=0
        if self.smiles_dict is not None:
            # read the smile data
            smile = self.smiles_dict[pocket]
    
            # convert the smiles to integers according to vocab
            smile = self.vocab.tokenize_smiles(smile)

            if len(smile)>vocab_len:
                vocab_len=len(smile)

            data.y = smile
            data.target = torch.tensor(smile[1:], dtype=torch.long)
            data.input = torch.tensor(smile[:-1], dtype=torch.long)


        # save the pocket name in data
        data.pocket_name = pocket   
        data.pocket_path = pocket_dir
        data.edge_index_first=edge_index_first
        data.edge_index_sequence=edge_index_sequence
        data.vocab_len=vocab_len
        hgnn_data.vocab_len=vocab_len


        sequence_output= get_protein_sequence_from_pdb(pocket_dir)
        data.sequence = sequence_output
        data.sequence = sequence_output


        return data, hgnn_data
        
def initial_vocab(vocab,vocab_path):
    if vocab == 'char':
        vocab = utils.CharVocab(vocab_path)
    elif vocab == 'selfies':
        vocab = utils.SELFIESVocab(vocab_path)
    elif vocab == 'regex':
        vocab = utils.RegExVocab(vocab_path)
    elif vocab=="moses":
        pass
    else:
        raise ValueError("invalid vocab value.")
    
    return vocab
