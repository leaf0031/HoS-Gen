from rdkit.Chem.rdMolAlign import CalcRMS
import json

json_file='dock_file_save/rl/2026_05_16_23_1088538616_1500/dock_result2/dock_dict.json'


dict_path='./smiles_pdb/crossdocked/2025_06_18_20_7057084/dock_protein_center/dock_'

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File {json_file} not found.")
    data = {}

affinity_values = {}
for key, values in data.items():
    
    for record in values:
        if record.get('mode_id') == 0:
            affinity_values[key] = record.get('affinity', None)  
            # print( affinity_values[key])
            break
affinity_list = [value for value in affinity_values.values() if value is not None]

if affinity_list:
    average_affinity = sum(affinity_list) / len(affinity_list)
    print(f"Average Affinity: {average_affinity:.4f}")
else:
    print("No valid affinity values found.")



