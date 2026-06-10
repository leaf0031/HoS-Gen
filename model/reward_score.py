import os
import subprocess
import tempfile
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors,AllChem,DataStructs
from Bio.PDB.PDBParser import PDBParser
from openbabel import pybel

class RewardScore:
    def __init__(self,device):
        self.device = device
        self.qed_base = 0.65 #0.7
        self.sa_base =  0.8 #0.85
        self.docking_base = -7.0
        reward_config = {
                'mw': 1.0
            }
        self.config = reward_config
        self.mw_min = 300
        self.mw_max = 600
        self.mw_sigma = 50
        self.batch_fp_list = []
        self.centroid = None
        self.receptor_pdbqt = None

    def reset_batch_memory(self):
        self.batch_fp_list = []
        self.centroid = None
        self.receptor_pdbqt = None
    
    def calculate_mw_score(self,mw):
        if self.mw_min <= mw <= self.mw_max:
            return 1.0 
        elif mw < self.mw_min:
            return np.exp(-((mw - self.mw_min)**2) / (2 * self.mw_sigma**2))
        else: # mw > self.mw_max
            return np.exp(-((mw - self.mw_max)**2) / (2 * self.mw_sigma**2))

    def calculate_center(self):

        parser = PDBParser()
        structure = parser.get_structure("pdbqt", self.receptor_pdbqt)
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords.append(atom.get_coord())
        coords = np.array(coords)
        center_of_mass = np.mean(coords, axis=0)
        center_of_mass = center_of_mass.astype(float)
        return center_of_mass
        
    def _run_fast_docking(self, smi, pocket_path):
        if self.receptor_pdbqt is None:
            fd, temp_pdbqt_path = tempfile.mkstemp(suffix='.pdbqt', prefix='receptor_')
            os.close(fd)
            try:
                rec = (f"python3 ./AutoDockTools_py3/AutoDockTools/Utilities24/prepare_receptor4.py -r {pocket_path} -o {temp_pdbqt_path} -A hydrogens")
                subprocess.run(rec, shell=True, check=True,capture_output=True)
                self.receptor_pdbqt = temp_pdbqt_path
            except subprocess.CalledProcessError as e:
                print('Pocket conversion to pdbqt failed')
                os.remove(temp_pdbqt_path)

        affinity = 0.0
        with tempfile.TemporaryDirectory() as tmpdirname:
            lig_pdb = os.path.join(tmpdirname, 'lig.pdb')
            lig_pdbqt = os.path.join(tmpdirname, 'lig.pdbqt')
            out_pdbqt = os.path.join(tmpdirname, 'out.pdbqt')
            try:
                mol = pybel.readstring("smi", smi)
                mols = mol.OBMol.Separate()
                mol = pybel.Molecule(mols[0])
                for imol in mols:
                    imol = pybel.Molecule(imol)
                    if len(imol.atoms) > len(mol.atoms):
                        mol = imol

                mol.addh()
                mol.make3D(forcefield='mmff94', steps=100)
                mol.localopt()
                mol.write(format='pdb', filename=str(lig_pdb), overwrite=True)
                
                cmd_prep = f"python3 /home/xxr/AutoDockTools_py3/AutoDockTools/Utilities24/prepare_ligand4.py -l {lig_pdb} -o {lig_pdbqt}"
                return_code = os.system(cmd_prep)
                if return_code != 0:
                    raise RuntimeError(f"Command execution failed, return code: {return_code}")
            
                if self.centroid is not None:
                    cx, cy, cz = self.centroid
                else:
                    self.centroid = self.calculate_center()
                    cx, cy, cz = self.centroid
                cmd_dock = f"""/home/xxr/qvina2.1 --receptor {self.receptor_pdbqt} --ligand {lig_pdbqt} \
                            --center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} \
                            --size_x 25 --size_y 25 --size_z 25 --exhaustiveness 4 --out {out_pdbqt}"""
                proc = subprocess.Popen(cmd_dock, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, _ = proc.communicate()
                
                output_str = stdout.decode('utf-8', errors='ignore')
                match = re.search(r'^\s*1\s+([-+0-9.]+)', output_str, re.MULTILINE)
                if match:
                    affinity = float(match.group(1))
            except Exception as e:
                print('docking error: ',e)
                return 0.0 
        return affinity

    def calculate_score(self, qed, sas_0_1, mol, smiles, pocket_path):


        mw = Descriptors.MolWt(mol)
        mw_score = self.calculate_mw_score(mw)

        dock_diff = 0
        affinity = self._run_fast_docking(smiles, pocket_path)
        print(affinity)
        if affinity < 0: 
            dock_diff = max(0.0, self.docking_base - affinity)

        # score = reward * self.config['composite_reward'] + sas_0_1 * self.config['sa_score'] 
        # score = qed * self.config['qed'] + docking_reward
        score = dock_diff + qed + mw_score
        
        return score
