import torch
import yaml
from tqdm import tqdm  
import pickle
from extract_electrostatic import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'bert-base'
TOKENIZER_VOCAB = 'iupac'
INPUT_FILE = 'case/7kzh_pocket2/7kzh_pocket2.yaml'  
pocket_dir = 'case/7kzh_pocket2/'
OUTPUT_DIR = '/home/data/elec/'  

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, 'r') as f:
    protein_sequences = yaml.safe_load(f)

print(f"Starting to extract features from {len(protein_sequences)} protein sequences...")

for name, seq in tqdm(protein_sequences.items(), desc="Processing progress"):
    try:
        base_name = os.path.basename(name)
        sub_dir = os.path.dirname(name)
        input_file = os.path.join(pocket_dir, name)
        output_sub_dir = os.path.join(OUTPUT_DIR, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)
        elec_extract = ElectrostaticGridGenerator()    #
        electrostatic_features = []

        grid_data = elec_extract.generate_electrostatic_grid(input_file, grid_dims=(129,129,129))
        grid_tensor = elec_extract.get_grid_tensor(normalize=True).unsqueeze(0)  

        file_name = os.path.splitext(base_name)[0] + '.pkl'
        feature_file = os.path.join(output_sub_dir, file_name)
   
        with open(feature_file, 'wb') as f:
            pickle.dump(grid_tensor, f)
            
    except Exception as e:
        print(f"Error processing {name}: {e}")

print(f"Feature extraction completed! A total of {len(protein_sequences)} sequences were processed.")
print(f"Feature file saved at: {OUTPUT_DIR}")

