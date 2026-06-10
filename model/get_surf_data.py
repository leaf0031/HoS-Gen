import sys
sys.path.append('')
import torch
import yaml
from tqdm import tqdm  
import pickle
from utils.get_surface import construct_surface
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'bert-base'
TOKENIZER_VOCAB = 'iupac'
INPUT_FILE = 'case/7kzh_pocket2/7kzh_pocket2.yaml' 
pocket_dir = 'case/7kzh_pocket2/'
OUTPUT_DIR = './data/surf/'  
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, 'r') as f:
    protein_sequences = yaml.safe_load(f)

print(f"Starting to extract features of {len(protein_sequences)} protein sequences...")

for name, seq in tqdm(protein_sequences.items(), desc="Processing progress"):
    try:
        base_name = os.path.basename(name)
        sub_dir = os.path.dirname(name)
        input_file = os.path.join(pocket_dir, name)

        output_sub_dir = os.path.join(OUTPUT_DIR, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        xyz, normal, curvature, atom, type = construct_surface(input_file)

        file_name = os.path.splitext(base_name)[0] + '.pkl'
        feature_file = os.path.join(output_sub_dir, file_name)
        print(feature_file)

        save_data = {
            'xyz': xyz,
            'normal': normal,
            'curvature': curvature,
            'atom': atom,
            'type': type
        }

        with open(feature_file, 'wb') as f:
            pickle.dump(save_data, f)
            
    except Exception as e:
        print(f"Error processing {name}: {e}")

print(f"Feature extraction completed! A total of {len(protein_sequences)} sequences were processed.")
print(f"Feature file saved at: {OUTPUT_DIR}")

