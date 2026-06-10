import os
import subprocess
from tqdm.auto import tqdm
import yaml
from typing import List, Optional

def convert_pdb_to_mol2(pdb_path: str, mol2_path: str) -> bool:
    try:
        cmd = f"obabel {pdb_path} -O {mol2_path}"
        subprocess.run(cmd, shell=True, check=True, 
                      stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        return os.path.exists(mol2_path)
    except subprocess.CalledProcessError as e:
        print(f"OpenBabel conversion failed {pdb_path}: {e.stderr.decode().strip()}")
        return False

def get_unique_filename(filepath: str) -> str:
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{ext}"
        counter += 1
    return filepath

def process_files(
    source_dir: str,
    target_pockets: Optional[List[str]] = None,
    pdb_suffix: str = "_pocket10.pdb"
) -> dict:
    """
    Processing PDB files and converting to MOL2 format

    Args:
    source_dir: Input directory (containing PDB files)
    target_pockets: List of pockets to process (None means process all)
    pdb_suffix: PDB file suffix matching pattern
    Returns:
    Statistics dictionary {'success': number of successes, 'failed': number of failures, 'failed_list': list of failed files}
    """
    stats = {'success': 0, 'failed': 0, 'failed_list': []}
    
    for root, _, files in tqdm(list(os.walk(source_dir)), desc="Processing"):
        for file in files:
            if not file.endswith(pdb_suffix):
                continue
                
            pocket_name = file.split('_rec_')[0] if '_rec_' in file else file.replace(pdb_suffix, "")

            if target_pockets and pocket_name not in target_pockets:
                continue

            pdb_path = os.path.join(root, file)
            mol2_file = file.replace('.pdb', '.mol2')
            mol2_path = os.path.join(root, mol2_file)
   
            mol2_path = get_unique_filename(mol2_path)
            
            if convert_pdb_to_mol2(pdb_path, mol2_path):
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['failed_list'].append(pdb_path)
    
    return stats

def load_yaml_targets(yaml_path: str) -> List[str]:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return list(config.keys())

if __name__ == '__main__':
    file_path = './data_test/test_pdb/1c8k_pocket.pdb'
    out_path = './data_test/test_pdb/1c8k_pocket.mol2'
    convert_pdb_to_mol2(file_path, out_path)
    