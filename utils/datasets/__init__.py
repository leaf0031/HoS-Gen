import torch
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset
import random


def get_dataset(config, *args, **kwargs):
    root = config['path']  
    dataset = PocketLigandPairDataset(root, *args, **kwargs)    

    return dataset
