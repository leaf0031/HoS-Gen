# building surface using dmasif
import numpy as np
from Bio.PDB import PDBParser
from utils.helper import *
from utils.Arguments import parser
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
from utils.geometry_processing import (
    curvatures,
    atoms_to_points_normals,
)


def construct_surface(pdb_root, args=None):
    args = parser.parse_args()
    atom_coords, atom_types = parse_pdb_atoms(pdb_root)
    atom = torch.from_numpy(atom_coords).cuda()
    atom_type = torch.from_numpy(atom_types).cuda()

    atom_batch = torch.zeros_like(atom)[:, 0]
    atom_batch = atom_batch.int()

    xyz, normal, batch = atoms_to_points_normals( 
        atom,
        atom_batch,
        atomtypes=atom_type,
        resolution=args.resolution,
        sup_sampling=args.sup_sampling,
        distance=args.distance,
    )

    P_curvatures = curvatures(
        xyz,
        triangles=None if args.use_mesh else None,
        normals=None if args.use_mesh else normal,
        scales=args.curvature_scales,
        batch=batch,
    )

    return xyz.cpu().numpy().astype('float32'), \
        normal.cpu().numpy().astype('float32'), \
        P_curvatures.cpu().numpy().astype('float32'), \
        atom.cpu().numpy().astype('float32'),\
        atom_type.cpu().numpy().astype('float32')



def get_kpconv_batch(surf_xyz, config, surf_features, sampleDl=0.1,verbose=0):
    """
    Generate the batch required for KPConv from surface point clouds
    :param surf_xyz: surface point coordinates (N, 3)
    :param surf_features: surface features (N, C)
    :param config: configuration file (includes number of layers, sampling steps, neighborhood parameters, etc.)
    :return: batch dictionary for KPConv
    """
    surf_features=surf_features.detach().cpu().numpy().astype(np.float32)
    points = []
    pools = []
    dl = config['first_subsampling_dl']
    points.append(surf_xyz)
    
    for i in range(config['num_layers']):
        if surf_features is not None:
            result = cpp_subsampling.subsample(points[i], features=surf_features, sampleDl=dl * (2 ** i), verbose=verbose)
            if isinstance(result, (tuple, list)):
                if len(result) == 2:
                    subsampled_points, surf_features = result
                else:
                    subsampled_points = result[0]
                    if len(result) > 1:
                        surf_features = result[1]
            else:
                subsampled_points = result
        else:
            subsampled_points = cpp_subsampling.subsample(
                points[i],
                sampleDl=dl * (2 ** i),
                verbose=verbose
            )

        pool_ind = get_pool_indices(points[i], subsampled_points)
        points.append(subsampled_points)
        pools.append(pool_ind)

    neighbors = []
    for i in range(config['num_layers']):
        radius = config['conv_radius'] * dl * (2 ** i)
        neighb_inds = batch_neighbors_kpconv(points[i], points[i], 
                                            torch.tensor([len(points[i])]), 
                                            torch.tensor([len(points[i])]), 
                                            radius, config['num_kernel_points'])
        neighbors.append(neighb_inds)

    batch = {
        "points": points,
        "neighbors": neighbors,
        "pools": pools,
        "upsamples": [],  
        "stack_lengths": [torch.tensor([len(surf_xyz)])],
        "embedding": surf_features
    }
    return batch

def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)
    
def parse_pdb_atoms(pdb_path):
    parser = PDBParser(QUIET=True)  
    structure = parser.get_structure("protein", pdb_path)
    
    atoms = []
    atom_types = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    for atom in residue:
                        atoms.append(atom.get_coord())
                        elem = atom.element
                        type_map = {"C": 0, "N": 1, "O": 2, "S": 3}
                        atom_types.append(type_map.get(elem, 4))  
    
    return np.array(atoms, dtype=np.float32), np.array(atom_types, dtype=np.int32)

def get_pool_indices(original_points, subsampled_points, tol=1e-6):
    if isinstance(original_points, torch.Tensor):
        original_points = original_points.cpu().numpy()
    if isinstance(subsampled_points, torch.Tensor):
        subsampled_points = subsampled_points.cpu().numpy()
    
    pool_ind = []
    for p in subsampled_points:
        dist = np.linalg.norm(original_points[:, :3] - p[:3], axis=1)
        idx = np.argmin(dist)
        pool_ind.append(idx)
    
    return np.array(pool_ind, dtype=np.int32)