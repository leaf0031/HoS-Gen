from Bio import PDB
import numpy as np
import torch

# Convert to edge index format
def convert_to_edge_index(hyperedges):
    edge_index = []

    # Check the hyperedge matrix row by row
    for i in range(hyperedges.shape[0]):
        # Get all amino acids connected to the current amino acid i
        connected_nodes = np.where(hyperedges[i] > 0)[0]

        # For each node connected to i, add an edge
        for node in connected_nodes:
            edge_index.append((i, node))  # Here, connecting node i and node

    return torch.tensor(edge_index, dtype=torch.long).t()  # Transpose to the edge_index format

# Extract amino acid information from the structure
# Amino acids and CA atom coordinates
def extract_amino_acids(structure, pocket_path):
    amino_acids = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                # residue output format: <Residue OXY het=H_OXY resseq=150 icode= >
                # PDB.Polypeptide.is_aa(residue) gets amino acid residues
                # Get amino acid coordinate information (using alpha carbon atom)
                # Check if it is a standard amino acid residue and not a hetero residue
                if PDB.Polypeptide.is_aa(residue):
                    # Check if it is a hetero residue
                    if residue.id[0] == ' ':
                        try:
                            amino_acids[(residue.parent.id, residue.id)] = [residue, residue['CA'].get_coord()]  # Maybe model.id should also be added, residue is indispensable
                        except:
                            print(pocket_path)
    return amino_acids

def build_vector_features(amino_acids,pdb_file_path):
   
    model = list(amino_acids.items())[0][1][0].parent.parent
   
    dssp = PDB.DSSP(model, pdb_file_path, dssp='/home/.conda/envs/HoS_Gen/bin/mkdssp')

    vertex_feature_vectors = {}
    
    # Construct the feature vector for the first vertex, DSSP calculated 8 types of secondary structures, one-hot
    # Secondary structures: H α-helix, B Isolated β-bridge residue, E Strand, G 3-10 helix, I Π-helix, T Turn, S Bend, - Other
    second_structure_labels = "HBEGITS-"

    
    # Obtain types of amino acids
    amino_acid_type_labels = "ACDEFGHIKLMNPQRSTVWY"

    # The statistical properties of chemical elements can include the relationships between chemical elements contained in amino acids, such as the number of benzene rings in amino acids, the number of hydroxyl groups, etc.
    # Collect the statistical properties of chemical elements for each amino acid
    chemical_element_statistics_labels = ["benzene_ring", "hydroxyl", "thiol", "phosphate"]

    for amino_acid_id, amino_acid in amino_acids.items():
        amino_acid_features = dssp[amino_acid_id]

        second_structure = np.zeros((1, 8))
        second_structure[0, second_structure_labels.index(amino_acid_features[2])] = 1

        spatial_features = np.array([[amino_acid_features[3], amino_acid_features[4], amino_acid_features[5]]])

        amino_acid_type = np.zeros((1, 20))
        amino_acid_type[0, amino_acid_type_labels.index(PDB.Polypeptide.protein_letters_3to1.get(amino_acid[0].resname))] = 1

        vertex_feature_vectors[amino_acid_id] = np.hstack((second_structure, spatial_features, amino_acid_type))

    
    return vertex_feature_vectors


# Construct the first hyperedge set to represent the spatial relationships between amino acids
def build_first_hyperedges(amino_acids, vertex_feature_vectors, k_neig, threshold):
    n = len(amino_acids)
    adjacency_matrix = None


    space_structure_hyperedges = np.zeros((n, n))
    distances = np.zeros((n, n))   
    sequence_structure_hyperedges = np.zeros((n, n))

    # Hyperedge of spatial structure
    for i, amino_acid_i in enumerate(amino_acids):
        for j, amino_acid_j in enumerate(amino_acids):  
            distances[i][j] = np.linalg.norm(np.array(amino_acids[amino_acid_i][1]) - np.array(amino_acids[amino_acid_j][1]))
            if distances[i][j] < threshold:  

                space_structure_hyperedges[j][i] =1
    adjacency_matrix = space_structure_hyperedges
    
    #knn hyperedge
    for center_idx in range(n):
        distances[center_idx, center_idx] = 0
        dis_vec = distances[center_idx]                
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()   
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx    

        for node_idx in nearest_idx[:k_neig]:
            sequence_structure_hyperedges[node_idx, center_idx] = 1

    adjacency_matrix = np.hstack((adjacency_matrix, sequence_structure_hyperedges))

    return space_structure_hyperedges, sequence_structure_hyperedges, adjacency_matrix

def obtian_edge(adjacency_matrix):
    adjacency_matrix = torch.from_numpy(adjacency_matrix)
    row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
    edge_index = torch.stack([row_indices, col_indices], dim=0)  
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    return edge_index


def pocket_hypergraph(pocket_path, threshold):
    protein_structure = PDB.PDBParser().get_structure("protein_structure", pocket_path)
    amino_acids=extract_amino_acids(protein_structure,pocket_path)
    vertex_feature_vectors = build_vector_features(amino_acids, pocket_path)
    node_features=torch.tensor(list(vertex_feature_vectors.values()),dtype=torch.float32).squeeze()
    k_neig=5
    space_edge, sequence_edge, first_edge= build_first_hyperedges(amino_acids, vertex_feature_vectors, k_neig, threshold)
    edge_index_space=obtian_edge(space_edge)
    # edge_index_space=space_edge
    edge_index_sequence=obtian_edge(sequence_edge)
    edge_index_first=obtian_edge(first_edge)
    return node_features, edge_index_space, edge_index_sequence, edge_index_first

