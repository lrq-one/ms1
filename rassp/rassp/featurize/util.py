# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Feature engineering utilities that transform molecules/spectra into model-ready tensors.

import numpy as np

# Function overview: get_nos_coords handles a specific workflow step in this module.
def get_nos_coords(mol, conf_i):
    conformers = mol.GetConformers()
    if len(conformers) == 0:
        raise ValueError("mol has no conformers")
    if conf_i < 0 or conf_i >= len(conformers):
        raise IndexError(f"conf_i={conf_i} out of range for {len(conformers)} conformers")
    conformer = conformers[conf_i]
    coord_objs = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coords = np.array([(c.x, c.y, c.z) for c in coord_objs])
    atomic_nos = np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)
    return atomic_nos, coords

# Function overview: get_nos handles a specific workflow step in this module.
def get_nos(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)

# Function overview: get_formula handles a specific workflow step in this module.
def get_formula(mol):
    """
    Return a dictionary of atomic_number -> count for atoms in mol.
    """
    out = {}
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        out[atomic_num] = out.get(atomic_num, 0) + 1
    return out

# Function overview: fast_multi_onehot handles a specific workflow step in this module.
def fast_multi_onehot(x, oh_offsets, out_array, accum=False):
    """
    Fast structured one-hot encoder.

    x: N x C integer matrix
    oh_offsets: length-C offsets into out_array columns
    out_array: N x D output matrix to write into
    accum: if True, writes cumulative one-hot [1,1,1,0,...] up to v
    """
    for row_i, row in enumerate(x):
        for col_i, value in enumerate(row):
            v = int(value)
            offset = int(oh_offsets[col_i])
            if accum:
                out_array[row_i, offset: offset + v] = 1
            else:
                out_array[row_i, offset + v] = 1

# Function overview: get_subset_peaks_from_formulae handles a specific workflow step in this module.
def get_subset_peaks_from_formulae(all_formulae, all_formulae_peaks,
                                   vert_element_oh,
                                   atom_subsets):
    """
    Get the mass peaks associated with each atom subset by looking them up
    in all_formulae and associated peaks. 
     
     all_formulae: N x ELEMENT_N all possible formuale
     all_formuale_peaks: N x peak_sizes
     vert_element_oh: one-hot-encoded matrix of atom types, ATOM_N x ELEMENT_N
     atom_subsets : atom subsets, M x ATOM_N
    """
     
    FORMULAE_N, ELEMENT_N = all_formulae.shape
    ATOM_N, _ = vert_element_oh.shape
    assert vert_element_oh.shape[1] == ELEMENT_N
    SUBSET_N, _ = atom_subsets.shape
    assert atom_subsets.shape[1] == ATOM_N
    
    formulae_pos_lut = {tuple(e): i for i, e in enumerate(all_formulae)}

    atom_subsets_peaks = np.zeros((SUBSET_N, all_formulae_peaks.shape[1]), dtype=all_formulae_peaks.dtype)

    for s, atom_subset in enumerate(atom_subsets):
        formula = (vert_element_oh * atom_subset.astype(np.float32).reshape(-1, 1)).sum(axis=0)
        formula_int = tuple(formula.astype(int))
        formula_lut_idx = formulae_pos_lut.get(formula_int, None)
        if formula_lut_idx is not None:
            atom_subsets_peaks[s] = all_formulae_peaks[formula_lut_idx]

    return atom_subsets_peaks

# Function overview: get_subset_peaks_idx_from_formulae_fast handles a specific workflow step in this module.
def get_subset_peaks_idx_from_formulae_fast(all_formulae, 
                                            vert_element_oh,
                                            atom_subsets):
    """
    Get the mass peaks indices associated with each atom subset by looking them up
    in all_formulae and associated peaks. 

    This is like get_subset_peaks_from_formulae, except we use more numpy
    operations and compute an integer for the hash instead of 
    using the formula as a tuple. 

     
     all_formulae: N x ELEMENT_N all possible formuale
     all_formuale_peaks: N x peak_sizes
     vert_element_oh: one-hot-encoded matrix of atom types, ATOM_N x ELEMENT_N
     atom_subsets : atom subsets, M x ATOM_N
    """
     
    FORMULAE_N, ELEMENT_N = all_formulae.shape
    ATOM_N, _ = vert_element_oh.shape
    assert vert_element_oh.shape[1] == ELEMENT_N
    SUBSET_N, _ = atom_subsets.shape
    assert atom_subsets.shape[1] == ATOM_N

    vert_element_oh = vert_element_oh.astype(np.int32)
    # how many of each element: What is the max of that value for the formula
    max_formula = vert_element_oh.astype(np.int64).sum(axis=0)

    # positionally-encode each element type as a larger and larger
    # integer to compute a hash. So our hash is
    #     num_h * 1  +  num_C * MAX_NUM_H  +  num_N * (MAX_NUM_H * MAX_NUM_C) + ....
    # 
    vert_element_accum =  np.cumprod(max_formula + 1)
    vert_element_accum[1:] = vert_element_accum[:-1]
    vert_element_accum[0] = 1
    
    f_int_key = all_formulae.astype(np.int64) @ vert_element_accum
    
    formulae_pos_lut = {e : i for i, e in enumerate(f_int_key)}

    formula_lut_idx_all = np.zeros(atom_subsets.shape[0], dtype=np.int64)

    # compute the hashes
    formulae_ints = atom_subsets @ (vert_element_oh @ vert_element_accum)

    # do the lookups
    for formula_i, formula_int in enumerate(formulae_ints):
        formula_lut_idx_all[formula_i] = formulae_pos_lut.get(int(formula_int), -1)

    return formula_lut_idx_all
