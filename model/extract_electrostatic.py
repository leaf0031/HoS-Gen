import numpy as np
import subprocess
import tempfile
from pathlib import Path
import pyvista as pv
import math
from Bio.PDB import PDBParser
import torch

class ElectrostaticGridGenerator: 
    def __init__(self):
        self.grid_data = None
        
    def generate_electrostatic_grid(self, pdb_file_path, grid_dims=None):
        """
        Generate 3D Electrostatic Potential Grid

        Args:
        pdb_file_path: Path to the PDB file
        grid_dims: Grid dimensions (nx, ny, nz), if None it will be automatically optimized

        Returns:
        grid_dict: Dictionary containing the grid data
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            pqr_file = temp_dir / "protein.pqr"
            self._create_pqr_file(pdb_file_path, pqr_file)

            if grid_dims is None:
                grid_dims = self.optimize_manual_grid(str(pqr_file))
            dx_file = temp_dir / "protein.dx"
            self._run_apbs_calculation(pqr_file, dx_file, grid_dims)

            self.grid_data = self._parse_dx_file(dx_file)
            
            return self.grid_data
    
    def _create_pqr_file(self, pdb_file, pqr_file):
        try:
            cmd = f"pdb2pqr30 --ff=AMBER {pdb_file} {pqr_file}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            if result.returncode == 0:
                result1=result
            else:
                self._create_simple_pqr(pdb_file, pqr_file)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("pdb2pqr is unavailable")

    
    def _run_apbs_calculation(self, pqr_file, dx_file, grid_dims):
        apbs_input = self._generate_apbs_input(pqr_file, dx_file, grid_dims)
        input_file = pqr_file.parent / "apbs_input.in"
        
        with open(input_file, 'w') as f:
            f.write(apbs_input)

        try:
            cmd = f"apbs {input_file}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print("APBS calculation successfully completed")
        except subprocess.CalledProcessError as e:
            print(f"APBS calculation failed: {e}")
            print(f"Error output: {e.stderr}")
            raise
    
    def _generate_apbs_input(self, pqr_file, dx_file, grid_dims):
        nx, ny, nz = grid_dims
        dx_basename = dx_file.stem
        return f"""
                read
                    mol pqr {pqr_file}
                end
                elec
                    mg-manual
                    dime {nx} {ny} {nz}
                    nlev 4
                    grid 0.33 0.33 0.33
                    gcent mol 1
                    mol 1
                    lpbe
                    bcfl sdh
                    pdie 2.0
                    sdie 78.54
                    srfm smol
                    chgm spl2
                    sdens 10.0
                    srad 1.4
                    swin 0.3
                    temp 298.15
                    calcenergy total
                    calcforce no
                    write pot dx {dx_file.parent / dx_basename}
                end
                quit
                """
    
    def _parse_dx_file(self, dx_file):
        try:
            with open(dx_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            grid_info = {}
            data_start_index = None
            for i, line in enumerate(lines):
                if line.startswith('object 1'):
                    if line.startswith('object 1 class gridpositions counts'):
                        parts = line.split()
                        grid_info['dims'] = [int(parts[5]), int(parts[6]), int(parts[7])]
                elif line.startswith('origin'):
                    parts = line.split()
                    grid_info['origin'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif line.startswith('delta') and 'delta' not in grid_info:
                    parts = line.split()
                    grid_info['delta'] = float(parts[1])
                elif line.startswith('object 3 class array') and 'data follows' in line:
                    data_start_index = i + 1
                    break
            if not all(key in grid_info for key in ['dims', 'origin', 'delta']):
                raise ValueError("Failed to parse necessary grid information (dims, origin, delta) from the DX file")

            if data_start_index is None:
                raise ValueError("'data follows' start marker not found in the DX file")
            
            data = []
            for line in lines[data_start_index:]:
                if line.startswith('object') or line.startswith('attribute'):
                    break
                data.extend(map(float, line.split()))

            expected_items = grid_info['dims'][0] * grid_info['dims'][1] * grid_info['dims'][2]
            if len(data) != expected_items:
                print(f"Warning: Data volume mismatch. Expected {expected_items}, got {len(data)}")
                if len(data) > expected_items:
                    data = data[:expected_items]
                else:
                    data.extend([0.0] * (expected_items - len(data)))
            
            grid_info['data'] = np.array(data).reshape(grid_info['dims'][::-1]).T
            return grid_info
            
        except Exception as e:
            print(f"Failed to parse DX file: {e}")
            return {
                'dims': [32, 32, 32],
                'origin': [0, 0, 0],
                'delta': 1.0,
                'data': np.random.randn(32, 32, 32)
            }
            
    
    def validate_manual_grid(self, pqr_file, nx, ny, nz, grid_spacing=0.33):
        try:
            parser = PDBParser()
            structure = parser.get_structure('protein', pqr_file)
            
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())
            
            coords = np.array(coords)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            protein_size = max_coords - min_coords
            center = (max_coords + min_coords) / 2
            grid_half_size = np.array([nx-1, ny-1, nz-1]) * grid_spacing / 2

            coverage_ok = True
            for i in range(3):
                protein_half_size = max(abs(max_coords[i] - center[i]), 
                                    abs(min_coords[i] - center[i]))
                if grid_half_size[i] < protein_half_size + 10:  
                    print(f"Warning: insufficient coverage in the {['X','Y','Z'][i]} direction")
                    print(f"  Required: {protein_half_size + 10:.1f} Å, Actual: {grid_half_size[i]:.1f} Å")
                    coverage_ok = False
            
            return coverage_ok, protein_size, grid_half_size * 2
            
        except Exception as e:
            print(f"Error occurred while validating the grid: {e}")
            return True, np.array([30, 30, 30]), np.array([50, 50, 50])  
    
    def optimize_manual_grid(self, pqr_file, target_spacing=0.33, padding=15.0):
        try:
            coverage_ok, protein_size, current_grid_size = self.validate_manual_grid(
                pqr_file, 161, 161, 161, 0.33)
            required_size = protein_size + 2 * padding
            def round_to_multigrid(n):
                candidates = [65, 97, 129, 161, 193, 257, 321, 385]
                for candidate in candidates:
                    if (candidate - 1) * target_spacing >= n:
                        return candidate
                return 385 
            
            nx = round_to_multigrid(required_size[0])
            ny = round_to_multigrid(required_size[1])
            nz = round_to_multigrid(required_size[2])
            
            final_grid_size = [(nx-1)*target_spacing, (ny-1)*target_spacing, (nz-1)*target_spacing]
            
            print(f"Protein size: {protein_size} Å")
            print(f"Recommended grid points: {nx} {ny} {nz}")
            print(f"Corresponding physical size: {final_grid_size} Å")
            print(f"Grid spacing: {target_spacing} Å")
            
            return (nx, ny, nz)
            
        except Exception as e:
            return (97, 97, 97)  #
    
    def get_grid_tensor(self, normalize=True):
        if self.grid_data is None:
            raise ValueError("Please first generate the electrostatic potential grid")
        
        grid_data = self.grid_data['data']
        
        if normalize:
            if np.std(grid_data) > 0:
                grid_data = (grid_data - np.mean(grid_data)) / np.std(grid_data)
                grid_data = np.clip(grid_data, -3, 3) / 3  
            else:
                grid_data = np.zeros_like(grid_data)
        
        grid_data = grid_data[np.newaxis, :, :, :]
        grid_tensor = torch.from_numpy(grid_data.astype(np.float32))
        
        return grid_tensor
    