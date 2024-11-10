import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    profile_distances = matrix_profile['mp']
    profile_indices = matrix_profile['mpi']

    for _ in range(top_k):
        motif_idx = np.argmin(profile_distances)
        motif_dist = profile_distances[motif_idx]

        motifs_idx.append((motif_idx, profile_indices[motif_idx]))
        motifs_dist.append(motif_dist)
        
        profile_distances = apply_exclusion_zone(profile_distances, motif_idx, matrix_profile['excl_zone'], np.inf)
        

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }