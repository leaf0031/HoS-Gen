from openbabel import pybel
import os
import subprocess
import yaml

def smi_pdb(smi,save_path):
    try:
        mol = pybel.readstring("smi", smi)
        mols = mol.OBMol.Separate()

        # print(pybel.Molecule(mols))

        mol = pybel.Molecule(mols[0])
        for imol in mols:
            imol = pybel.Molecule(imol)
            if len(imol.atoms) > len(mol.atoms):
                mol = imol

        mol.addh()
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='pdb', filename=str(save_path), overwrite=True)
        return 1
    except:
        print(f"Tranformation of {smi} failed! ")
        return 0


def pdb_to_pdbqt(input_pdb, output_pdbqt):
    try:
        command = (
            f'python3 /home/xxr/AutoDockTools_py3/AutoDockTools/Utilities24/prepare_ligand4.py '
            f'-l {input_pdb} '
            f'-o {output_pdbqt}'
        )
        return_code = os.system(command)
        if return_code != 0:
            raise RuntimeError(f"Command execution failed, return code: {return_code}")
            
        return True
    except Exception as e:
        print(f"Conversion of {input_pdb} failed! Error: {str(e)}")
        return False

def get_smi(config_dir):
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)
    return list(config.keys())



openbable_dir= './evaluation_dock/smi2pdbqt.yaml'
with open(openbable_dir, 'r') as f:
    config = yaml.full_load(f)


pocket_path = config['pocket_path']
smiles_yaml = config['smiles_yaml']
save_smiles_pdb = config['smiles_pdb']
save_smiles_pdbqt = config['smiles_pdbqt']
dataset = config['dataset']

if dataset == 'pdbbind':
    pocket_path = './data_pdbbind/test_64.yaml'
else:
    pocket_path = './data_crossdocked/test.yaml'


if not os.path.exists(save_smiles_pdb):
    os.makedirs(save_smiles_pdb)

if not os.path.exists(save_smiles_pdbqt):
    os.makedirs(save_smiles_pdbqt)


with open(pocket_path, 'r') as f:
    pocket_dict = yaml.full_load(f)
pocket_names=list(pocket_dict.keys())


list_error=[]
error_2pdbqt = []

for index, pocket_item in enumerate(pocket_names):

    if dataset=='crossdocked':
        pocket_item = os.path.splitext(os.path.basename(pocket_item))[0]

    pocket_smiles_path = os.path.join(smiles_yaml, pocket_item)+ '_sampled_temp1.yaml'
    smiles = get_smi(pocket_smiles_path)

    for index, smile in enumerate(smiles):

        each_save_pdb = os.path.join(save_smiles_pdb, pocket_item) +'_' +str(index) + '.pdb'
        each_save_pdbqt = os.path.join(save_smiles_pdbqt, pocket_item) +'_' +str(index) + '.pdbqt'
        result = smi_pdb(smile, each_save_pdb)
        if result==0:
            error_item = pocket_item + '_' + str(index)
            list_error.append(error_item)
        else:
            result_pdbqt = pdb_to_pdbqt(each_save_pdb,each_save_pdbqt)
            if not result:
                error_item = pocket_item + '_' + str(index)
                error_2pdbqt.append(error_item)

save_error_list = config['error_list']
with open(save_error_list +'error_2pdb.yaml', 'w') as f:
    yaml.dump(list_error, f)

with open(save_error_list +'error_2pdbqt.yaml', 'w') as f:
    yaml.dump(error_2pdbqt, f)

with open(save_error_list +'smi2pdbqt.yaml', 'w') as f:
    yaml.dump(config, f)
