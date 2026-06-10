
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa, three_to_index, index_to_one

def get_protein_sequence_from_pdb(pdb_file_path):
    """
        Extract the amino acid sequence of a protein from a PDB file.
        pdb_file_path (str): Path to the PDB file.
        Returns:
        str: The extracted amino acid sequence (single-letter code).
        If no protein chain is found in the PDB file, returns an empty string.
    """
    # Create a PDB parser object
    parser = PDB.PDBParser(QUIET=True)  # QUIET=True to avoid printing unnecessary warnings

    try:
        structure = parser.get_structure("structure", pdb_file_path)
    except:
        return ""  # Return empty if parsing fails

    # Initialize an empty list to store the sequence
    full_sequence = []
    for model in structure:
        for chain in model:
            # Check if this chain is a protein chain
            if any(residue.has_id('CA') for residue in chain):
                for residue in chain:
                    # Check if the residue is an amino acid
                    if not is_aa(residue, standard=False):
                        continue  # Skip non-amino acids
                    res_name = residue.get_resname().strip()  # Remove possible spaces
                    try:
                        one_letter = index_to_one(three_to_index(res_name.upper()))
                        full_sequence.append(one_letter)
                    except KeyError:
                        # Print warning for non-standard amino acids
                        print(f"Unknown residue (non-standard): {res_name}")

    # If the list is not empty, join it into a string
    if full_sequence:
        return ''.join(full_sequence)
    else:
        return "No protein chain or amino acids found in this PDB file."
