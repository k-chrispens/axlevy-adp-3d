import torch.nn as nn
import numpy as np
import torch
from Bio import SeqIO
import mrcfile
import sys
from chroma import constants


ATOM_TO_Z = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'S':16.0}


def letter_to_n_atoms(letter):
    return len(constants.AA_GEOMETRY[constants.AA20_1_TO_3[letter]]['atoms']) + 4


class RealisticSolver(nn.Module):
    def __init__(self, mrc_path, S, remove_oxt=True, normalization='constant', resolution=2.0, unpad_len=0, outdir=None):
        super(RealisticSolver, self).__init__()
        self.normalization = normalization
        self.resolution = resolution

        self.n_residues = S.shape[-1]
        letters = [constants.AA20[int(s)] for s in S[0]]
        
        self.indices_X_full_to_coords = []
        for i, letter in enumerate(letters):
            n_atoms = letter_to_n_atoms(letter)
            for j in range(n_atoms):
                self.indices_X_full_to_coords.append(i * 14 + j)

        self.n_atoms = len(self.indices_X_full_to_coords)
        amplitudes = []
        for letter in letters:
            atoms = constants.AA_GEOMETRY[constants.AA20_1_TO_3[letter]]['atoms']
            for atom in ['N', 'C', 'C', 'O']:
                amplitudes.append(ATOM_TO_Z[atom])
            for atom in atoms:
                amplitudes.append(ATOM_TO_Z[atom[0]])
        self.amplitudes = torch.tensor(amplitudes).float().cuda()
        self.sigma = resolution / (np.sqrt(2.) * np.pi)

        self.indices_coords_to_X = []
        idx = 0
        for letter in letters:
            for i in range(4):
                self.indices_coords_to_X.append(idx + i)
            idx += letter_to_n_atoms(letter)

        # 3d grid
        if unpad_len == 0:
            with mrcfile.open(mrc_path) as mrc:
                voxel_size = mrc.voxel_size['x'].item()
                self.density = mrc.data
                self.mrc_header = mrc.header
                nx = mrc.header['nx'].item()
                ny = mrc.header['ny'].item()
                nz = mrc.header['nz'].item()
        else:
            with mrcfile.open(mrc_path) as mrc:
                voxel_size = mrc.voxel_size['x'].item()
                density = mrc.data
                header = mrc.header
            density_unpad = density[unpad_len:-unpad_len, unpad_len:-unpad_len, unpad_len:-unpad_len]
            print(f"Saving {outdir}/gt_unpadded.mrc")
            with mrcfile.new(f"{outdir}/gt_unpadded.mrc", overwrite=True) as mrc:
                mrc.set_data(density_unpad)
            with mrcfile.open(f"{outdir}/gt_unpadded.mrc", mode='r+') as mrc:
                mrc.voxel_size = voxel_size
                cella = (header.cella['x'] - voxel_size * 2 * unpad_len, header.cella['y'] - voxel_size * 2 * unpad_len, header.cella['z'] - voxel_size * 2 * unpad_len)
                dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
                arr = np.rec.array(cella, dtype=dtype)
                mrc.header.cella = arr
                origin = (header.origin['x'] + voxel_size * unpad_len, header.origin['y'] + voxel_size * unpad_len, header.origin['z'] + voxel_size * unpad_len)
                dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
                arr = np.rec.array(origin, dtype=dtype)
                mrc.header.origin = arr
            with mrcfile.open(f"{outdir}/gt_unpadded.mrc") as mrc:
                self.density = mrc.data
                self.mrc_header = mrc.header
                nx = mrc.header['nx'].item()
                ny = mrc.header['ny'].item()
                nz = mrc.header['nz'].item()
        self.origin = torch.tensor([self.mrc_header['origin'][i] + voxel_size * (self.mrc_header['n' + i] // 2) for i in ['x', 'y', 'z']]).cuda()
        ax_x = torch.arange(-nx // 2, -nx // 2 + nx).float()
        ax_y = torch.arange(-ny // 2, -ny // 2 + ny).float()
        ax_z = torch.arange(-nz // 2, -nz // 2 + nz).float()
        pix_coords_Z, pix_coords_Y, pix_coords_X = torch.meshgrid(ax_z, ax_y, ax_x, indexing='ij')
        self.pix_coords_3d = torch.stack([pix_coords_X, pix_coords_Y, pix_coords_Z],
                                         dim=-1).reshape(-1, 3).cuda() * voxel_size
        self.n_pix = nx * ny * nz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.voxel_size = voxel_size

        # 3d grid in Fourier space
        f_ax_x = torch.arange(-(nx // 2), (nx + 1) // 2).float()
        f_ax_y = torch.arange(-(ny // 2), (ny + 1) // 2).float()
        f_ax_z = torch.arange(-(nz // 2), (nz + 1) // 2).float()
        zz, yy, xx = torch.meshgrid(f_ax_z, f_ax_y, f_ax_x, indexing='ij')
        self.f_pix_coords_3d = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).cuda() / voxel_size

    def X_to_coords(self, X):
        X_flat = X.reshape(X.shape[0], -1, 3)
        return X_flat[:, self.indices_X_full_to_coords]

    def coords_to_X(self, coords, all_atom=True):
        if all_atom:
            X = torch.zeros(1, self.n_residues, 14, 3).float().to(coords.device)
            X_flat = X.reshape(-1, 3)
            X_flat[self.indices_X_full_to_coords] = coords
            return X_flat.reshape(1, self.n_residues, 14, 3)
        else:
            X_flat = coords[self.indices_coords_to_X]
            return X_flat.reshape(1, self.n_residues, 4, 3)

    def get_mask(self, resolution_cutoff):
        return (torch.linalg.norm(self.pix_coords_3d, dim=-1).reshape(self.nz, self.ny, self.nx) <
                (self.nx * self.voxel_size ** 2 / (2. * resolution_cutoff)))

    def normalize(self, potential):
        if self.normalization == 'constant':
            return potential
        else:
            norm = torch.clone(torch.sqrt(torch.sum(potential ** 2)).detach()) * 1e-3
            return potential / norm

    def get_one_batch_f_density(self, X, f_potential_gt, resolution_cutoff, normalize=True, sampling_rate=1.):
        mask = self.get_mask(resolution_cutoff)
        mask *= (torch.bernoulli(torch.ones_like(mask) * sampling_rate) > 0.5)
        f_potential_gt_sampled = f_potential_gt.reshape(-1)[mask.cpu().reshape(-1)].cuda()
        f_pix_coords_3d_sampled = self.f_pix_coords_3d.reshape(-1, 3)[mask.reshape(-1)]
        coords = self.X_to_coords(X)
        f_potential_sampled = self.calculate_f_potential_one_batch(coords, f_pix_coords_3d_sampled).reshape(
            X.shape[0], -1)
        if normalize:
            f_potential_sampled = self.normalize(f_potential_sampled)
            f_potential_gt_sampled = self.normalize(f_potential_gt_sampled)
        return f_potential_sampled, f_potential_gt_sampled

    def get_full_f_density_per_batch(self, X, n_sampled_voxels, normalize=True):
        f_potential_full = []
        coords = self.X_to_coords(X)
        for i in range(((self.n_pix - 1) // n_sampled_voxels) + 1):
            i_min = i * n_sampled_voxels
            i_max = min((i + 1) * n_sampled_voxels, self.n_pix)
            sampled_indices = torch.arange(i_min, i_max)
            f_pix_coords_3d_sampled = self.f_pix_coords_3d[sampled_indices]
            assert (torch.linalg.norm(coords, dim=-1) < 1-6).sum() <= 1, f"{(torch.linalg.norm(coords, dim=-1) < 1-6).sum()} atoms at the origin"
            f_potential_sampled = self.calculate_f_potential_one_batch(coords, f_pix_coords_3d_sampled).reshape(-1)
            f_potential_full.append(f_potential_sampled.detach().cpu())
        f_potential_full = torch.cat(f_potential_full, 0).reshape(1, self.nz, self.ny, self.nx)
        return f_potential_full

    def calculate_f_potential_one_batch(self, coords, f_pix_coords_3d_sampled):
        '''
        coords: N, n_atoms, 3
        f_pix_coords_3d_sampled: n_sampled_voxels, 3
        '''
        dimension = coords.shape[-1]
        n_sampled_voxels = f_pix_coords_3d_sampled.shape[0]

        sigma = self.sigma
        f_sigma = torch.tensor([
            self.nx / (2. * np.pi * sigma),
            self.ny / (2. * np.pi * sigma),
            self.nz / (2. * np.pi * sigma),
        ]).float().cuda()
        alpha = torch.tensor([
            (2. * np.pi) / self.nx,
            (2. * np.pi) / self.ny,
            (2. * np.pi) / self.nz,
        ]).float().cuda()

        dist_sq = ((f_pix_coords_3d_sampled / f_sigma) ** 2 / 2.).sum(-1).reshape(1, n_sampled_voxels)  # [1, n_sampled_voxels]
        amplitude = (-dist_sq).exp() * self.amplitudes[..., None]  # [n_atoms, n_sampled_voxels]
        real_part = amplitude * torch.cos(-(f_pix_coords_3d_sampled * coords[..., None, :] * alpha).sum(-1))  # [N, n_atoms, n_sampled_voxels]
        imag_part = amplitude * torch.sin(-(f_pix_coords_3d_sampled * coords[..., None, :] * alpha).sum(-1))  # [N, n_atoms, n_sampled_voxels]
        f_potential = torch.view_as_complex(torch.cat([
            real_part[..., None],
            imag_part[..., None]
        ], -1)).sum(-2)  # [N, n_sampled_voxels]
        return f_potential