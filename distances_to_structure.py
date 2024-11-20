import argparse
import torch
import os
import numpy as np
from scipy.linalg import svd, inv
import pickle
from chroma import Protein
from chroma import Chroma
from chroma.layers.structure.rmsd import CrossRMSD

from src.plots import plot_metric


def main(args):
    pass
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    print("Saving arguments")
    args_dict = vars(args)
    with open(f"{args.outdir}/config.txt", 'w') as file:
        for key in sorted(args_dict.keys()):
            file.write(f"{key}: {args_dict[key]}\n")

    print("Loading Chroma model")
    if args.weights_backbone is not None and args.weights_design is not None:
        chroma = Chroma(weights_backbone=args.weights_backbone, weights_design=args.weights_design)
    else:
        chroma = Chroma()
    backbone_network = chroma.backbone_network
    design_network = chroma.design_network
    def multiply_R(Z, C): return backbone_network.noise_perturb.base_gaussian._multiply_R(Z, C)
    def multiply_R_inverse(X, C): return backbone_network.noise_perturb.base_gaussian._multiply_R_inverse(X, C)

    print("Initializing ground truth")
    protein = Protein.from_CIF(args.cif, device='cuda')
    X_gt, C_gt, S_gt = protein.to_XCS(all_atom=False)
    X_gt = X_gt[torch.abs(C_gt) == 1][None]
    S_gt = S_gt[torch.abs(C_gt) == 1][None]
    C_gt = C_gt[torch.abs(C_gt) == 1][None]
    X_gt -= X_gt.mean(dim=(0, 1, 2))
    n_residues = X_gt.shape[1]

    def X_to_distance_matrix(X):
        nodes = X[:, :, 1, :]  # [N, R, 3]
        distance_matrix = torch.linalg.norm(nodes[:, :, None] - nodes[:, None], dim=-1)
        return distance_matrix  # [N, R, R]

    distance_matrix_gt = X_to_distance_matrix(X_gt)  # [1, R, R]
    if args.n_distances == -1:
        mask_dist = np.ones(distance_matrix_gt.shape)
    else:
        idxs = torch.arange(n_residues).cuda()
        idxs = idxs[(C_gt == 1).reshape(-1)]
        mask_dist = np.zeros(distance_matrix_gt.shape)
        n_ones = 0
        while n_ones < args.n_distances:
            i = idxs[np.random.randint(len(idxs))]
            j = idxs[np.random.randint(len(idxs))]
            if i < j + 1:
                if mask_dist[0, i, j] < .5:
                    mask_dist[0, i, j] = 1.
                    n_ones += 1
    mask_dist = torch.tensor(mask_dist).float().cuda()
    Y_dist = mask_dist * distance_matrix_gt  # [1, R, R]

    def get_gradient_Z_dist(Z):
        if args.lr_distance > 0.:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _X = multiply_R(Z, C_gt)
                distance_matrix = X_to_distance_matrix(_X)  # [N, R, R]
                loss_dist = ((Y_dist - mask_dist * distance_matrix) ** 2).mean(dim=(1, 2)).sum()
                loss_dist.backward()
                grad_Z_dist = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_dist = torch.zeros(*Z.shape).float().to(X.device)
            loss_dist = torch.tensor([0.]).float().cuda()
        return grad_Z_dist, loss_dist

    print("Initializing backbone")
    C_gt = C_gt.expand(args.population_size, -1)
    S_gt = S_gt.expand(args.population_size, -1)
    if args.init_gt:
        X = torch.clone(X_gt).expand(args.population_size, -1, -1, -1)
    else:
        Z = torch.randn(args.population_size, *X_gt.shape[1:]).float().cuda()
        X = multiply_R(Z, C_gt)
    V_dist = torch.zeros_like(Z)

    def t_fn(epoch):
        if args.temporal_schedule == 'linear':
            return (-args.t + 0.001) * epoch / args.epochs + args.t
        elif args.temporal_schedule == 'sqrt':
            return (1.0 - 0.001) * (1. - np.sqrt(epoch / args.epochs)) + 0.001
        elif args.temporal_schedule == 'constant':
            return args.t
        else:
            raise NotImplementedError

    metrics = {'epoch': [], 'rmsd': [], 't': [], 'rmsd_ca': [], 'loss_dist': []}

    print("--- Optimization starts now ---")
    for epoch in range(args.epochs):
        t = torch.tensor(t_fn(epoch)).float().cuda()

        if args.use_diffusion:
            with torch.no_grad():
                X0 = backbone_network.denoise(X.detach(), C_gt, t)
        else:
            X0 = X
        Z0 = multiply_R_inverse(X0, C_gt)

        grad_Z_dist, loss_dist = get_gradient_Z_dist(Z0)
        V_dist = args.rho_distance * V_dist + args.lr_distance * grad_Z_dist

        Z0 = Z0 - V_dist

        if args.use_diffusion:
            tm1 = torch.tensor(t_fn(epoch + 1)).float().cuda()
            alpha, sigma, _, _, _, _ = backbone_network.noise_perturb._schedule_coefficients(tm1)
            X = multiply_R(alpha * Z0 + sigma * torch.randn_like(Z0), C_gt)
        else:
            X = multiply_R(Z0, C_gt)

        if (epoch + 1) % args.log_every == 0:
            rmsds = []
            rmsds_cas = []
            for i in range(args.population_size):
                rmsd, _ = CrossRMSD().pairedRMSD(
                    torch.clone(X[i, C_gt[0] == 1]).cpu().reshape(1, -1, 3),
                    torch.clone(X_gt[0, C_gt[0] == 1]).cpu().reshape(1, -1, 3),
                    compute_alignment=True
                )
                rmsd_ca, _ = CrossRMSD().pairedRMSD(
                    torch.clone(X[i, C_gt[0] == 1, 1, :]).cpu().reshape(1, -1, 3),
                    torch.clone(X_gt[0, C_gt[0] == 1, 1, :]).cpu().reshape(1, -1, 3),
                    compute_alignment=True
                )
                rmsds.append(rmsd.item())
                rmsds_cas.append(rmsd_ca.item())
            idx_best = np.argmin(rmsds)
            rmsd_best = rmsds[idx_best]
            rmsd_ca_best = rmsds_cas[idx_best]

            rmsd, X_aligned = CrossRMSD().pairedRMSD(
                torch.clone(X[idx_best]).cpu().reshape(1, -1, 3),
                torch.clone(X_gt[0]).cpu().reshape(1, -1, 3),
                compute_alignment=True
            )
            X_aligned = X_aligned.reshape(1, -1, 4, 3)

            metrics['epoch'].append(epoch)
            metrics['t'].append(t.item())
            metrics['loss_dist'].append(loss_dist.item())
            metrics['rmsd'].append(rmsds)
            metrics['rmsd_ca'].append(rmsds_cas)
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss Distance: {loss_dist.item():.4e}, RMSD: {rmsd_best:.2e}, RMSD CA: {rmsd_ca_best:.2e}")

    C_gt = C_gt[0:1]
    S_gt = S_gt[0:1]
    rmsd, X_aligned = CrossRMSD().pairedRMSD(
        torch.clone(X[idx_best]).cpu().reshape(1, -1, 3),
        torch.clone(X_gt[0]).cpu().reshape(1, -1, 3),
        compute_alignment=True
    )
    X_aligned = X_aligned.reshape(1, -1, 4, 3).cuda()

    print(f"Saving {args.outdir}/metrics.pkl")
    with open(f"{args.outdir}/metrics.pkl", 'wb') as file:
        pickle.dump(metrics, file)

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}.pdb")
    protein_out = Protein.from_XCS(X_aligned, C_gt, S_gt)
    protein_out.to_PDB(f"{args.outdir}/{args.outdir.split('/')[-1]}.pdb")

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}_gt.pdb")
    protein_out = Protein.from_XCS(X_gt, C_gt, S_gt)
    protein_out.to_PDB(f"{args.outdir}/{args.outdir.split('/')[-1]}_gt.pdb")

    for key in metrics.keys():
        if key != 'epoch':
            print(f"Saving {args.outdir}/{key}.png")
            plot_metric(metrics, key, f"{args.outdir}/{key}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--cif', type=str, required=True)

    # I/O parameters
    parser.add_argument('--n-distances', type=int, default=-1)
    parser.add_argument('--weights-backbone', type=str, default=None)
    parser.add_argument('--weights-design', type=str, default=None)

    # optimization parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--population-size', type=int, default=8)
    parser.add_argument('--lr-distance', type=float, default=0.01)
    parser.add_argument('--rho-distance', type=float, default=0.99)
    parser.add_argument('--temporal-schedule', type=str, default='sqrt')
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--use-diffusion', type=int, default=1)

    # initialization parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init-gt', type=int, default=0)

    # logging parameters
    parser.add_argument('--log-every', type=int, default=10)

    args = parser.parse_args()
    main(args)
