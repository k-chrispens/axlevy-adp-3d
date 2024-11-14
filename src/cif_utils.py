from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
import torch


def ma_cif_to_X(path, n_residues):
    dict = MMCIF2Dict(path)
    label_seq_id = np.array(dict['_atom_site.label_seq_id'])
    label_atom_id = np.array(dict['_atom_site.label_atom_id'])
    xs = np.array(dict['_atom_site.Cartn_x'])
    ys = np.array(dict['_atom_site.Cartn_y'])
    zs = np.array(dict['_atom_site.Cartn_z'])

    X = torch.zeros(1, n_residues, 4, 3).float()
    mask = torch.zeros(n_residues).float()

    for idx, element, x, y, z in zip(label_seq_id, label_atom_id, xs, ys, zs):
        if element == 'N':
            X[0, int(idx) - 1, 0] = torch.tensor([float(x), float(y), float(z)]).float()
        if element == 'CA':
            X[0, int(idx) - 1, 1] = torch.tensor([float(x), float(y), float(z)]).float()
        if element == 'C':
            X[0, int(idx) - 1, 2] = torch.tensor([float(x), float(y), float(z)]).float()
        if element == 'O':
            X[0, int(idx) - 1, 3] = torch.tensor([float(x), float(y), float(z)]).float()
        mask[int(idx) - 1] = 1.

    return X, mask.reshape(1, -1, 1, 1)