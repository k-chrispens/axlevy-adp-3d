import argparse
import torch
import os
import numpy as np
from scipy.linalg import svd, inv
import pickle
import time
from chroma import Protein
from chroma import Chroma
from chroma.layers.structure.rmsd import CrossRMSD

from src.plots import plot_metric


def main(args):
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

    print("Precomputing preconditioning matrices")
    mask_flat = torch.zeros(n_residues).float().cuda()
    mask_flat[::args.fix_every] = 1.
    mask_flat[(C_gt != 1).reshape(-1)] = 0.
    Y = X_gt[:, mask_flat > 0.5]

    def mR_fun(z):
        z = z.reshape(-1, n_residues, 4, 3)
        return multiply_R(z, C_gt.expand(z.shape[0], -1))[:, mask_flat > 0.5].reshape(z.shape[0], -1).permute(1, 0)

    Z = torch.eye(n_residues * 12).cuda()
    mR_mat = mR_fun(Z).cpu().numpy()

    U, S, Vh = svd(mR_mat)

    mR_mat_T = torch.tensor(mR_mat.T).float().cuda()
    S = torch.tensor(S).float().cuda()
    Vh = torch.tensor(Vh).float().cuda()
    STS = torch.zeros(Vh.shape[0]).float().cuda()
    STS[:U.shape[0]] = S ** 2
    Vh_T = torch.tensor(Vh.T).float().cuda()

    U_T_expanded = torch.tensor(U.T).float().cuda()[None].expand(args.population_size, -1, -1)
    Vh_expanded = Vh[None].expand(args.population_size, -1, -1)

    def get_gradient_Z_m(Z, t):
        if args.lr_model > 0.:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _Z = multiply_R_inverse(backbone_network.denoise(multiply_R(Z, C_gt), C_gt, t), C_gt)
                if args.preconditioning_model:
                    UTY = torch.bmm(U_T_expanded, Y.reshape(Y.shape[0], -1, 1)).reshape(Y.shape[0], -1)
                    Sm1UTY = UTY / S
                    loss_m = ((torch.bmm(Vh_expanded, _Z.reshape(Z.shape[0], -1, 1))[:, :U.shape[0],
                               0] - Sm1UTY) ** 2).sum()
                else:
                    loss_m = ((multiply_R(_Z, C_gt)[:, mask_flat > 0.5] - Y) ** 2).sum()
                loss_m.backward()
                grad_Z_m = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_m = torch.zeros(*Z.shape).float().to(X.device)
            loss_m = torch.tensor([0.]).float().cuda()
        return grad_Z_m, loss_m

    print("Initializing backbone")
    C_gt = C_gt.expand(args.population_size, -1)
    S_gt = S_gt.expand(args.population_size, -1)
    Y = Y.expand(args.population_size, -1, -1, -1)
    if args.init_gt:
        X = torch.clone(X_gt).expand(args.population_size, -1, -1, -1)
    else:
        Z = torch.randn(args.population_size, *X_gt.shape[1:]).float().cuda()
        X = multiply_R(Z, C_gt)
    V_m = torch.zeros_like(Z)

    def t_fn(epoch):
        if args.temporal_schedule == 'linear':
            return (-args.t + 0.001) * epoch / args.epochs + args.t
        elif args.temporal_schedule == 'sqrt':
            return (1.0 - 0.001) * (1. - np.sqrt(epoch / args.epochs)) + 0.001
        elif args.temporal_schedule == 'constant':
            return args.t
        else:
            raise NotImplementedError

    def backward_sde_one_step(X, C, t1, t2):
        t1 = torch.tensor(t1).cuda()
        t2 = torch.tensor(t2).cuda()
        X0_func = lambda _X, _C, t: backbone_network.denoise(_X, _C, t)
        dT = t2 - t1
        f, gZ = backbone_network.noise_perturb.reverse_sde(X, X0_func, C, t1, inverse_temperature=10.0, langevin_factor=2.0, langevin_isothermal=False)
        return X + dT * f + dT.abs().sqrt() * gZ

    metrics = {'epoch': [], 'rmsd': [], 't': [], 'rmsd_ca': [], 'loss_m': []}

    print("--- Optimization starts now ---")
    tic = time.time()
    for epoch in range(args.epochs):
        t = torch.tensor(t_fn(epoch)).float().cuda()

        Z0 = multiply_R_inverse(X, C_gt)
        grad_Z_m, loss_m = get_gradient_Z_m(Z0, t)
        V_m = args.rho_model * V_m + args.lr_model * grad_Z_m / torch.linalg.norm(grad_Z_m).detach()
        Z0 = Z0 - V_m
        _X = multiply_R(Z0, C_gt)
        tm1 = torch.tensor(t_fn(epoch + 1)).float().cuda()
        X = backward_sde_one_step(_X, C_gt, t, tm1)

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

            metrics['epoch'].append(epoch)
            metrics['t'].append(t.item())
            metrics['loss_m'].append(loss_m.item())
            metrics['rmsd'].append(rmsds)
            metrics['rmsd_ca'].append(rmsds_cas)
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss Model: {loss_m.item():.4e}, RMSD: {rmsd_best:.2e}, RMSD CA: {rmsd_ca_best:.2e}")

            print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}_{epoch:04}.pdb")
            protein_out = Protein.from_XCS(X[idx_best][None], C_gt[0:1], S_gt[0:1])
            protein_out.to_PDB(f"{args.outdir}/{args.outdir.split('/')[-1]}_{epoch:04}.pdb")

    metrics['total_time'] = time.time() - tic
    
    C_gt = C_gt[0:1]
    S_gt = S_gt[0:1]

    print(f"Saving {args.outdir}/metrics.pkl")
    with open(f"{args.outdir}/metrics.pkl", 'wb') as file:
        pickle.dump(metrics, file)

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}.pdb")
    protein_out = Protein.from_XCS(X[idx_best][None], C_gt, S_gt)
    protein_out.to_PDB(f"{args.outdir}/{args.outdir.split('/')[-1]}.pdb")

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}_gt.pdb")
    protein_out = Protein.from_XCS(X_gt, C_gt, S_gt)
    protein_out.to_PDB(f"{args.outdir}/{args.outdir.split('/')[-1]}_gt.pdb")

    for key in metrics.keys():
        if key != 'epoch' and key != 'total_time':
            print(f"Saving {args.outdir}/{key}.png")
            plot_metric(metrics, key, f"{args.outdir}/{key}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--cif', type=str, required=True)

    # I/O parameters
    parser.add_argument('--fix-every', type=int, default=2)
    parser.add_argument('--weights-backbone', type=str, default=None)
    parser.add_argument('--weights-design', type=str, default=None)

    # optimization parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--population-size', type=int, default=8)
    parser.add_argument('--lr-model', type=float, default=0.3)
    parser.add_argument('--rho-model', type=float, default=0.0)
    parser.add_argument('--temporal-schedule', type=str, default='linear')
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--preconditioning-model', type=int, default=1)
    parser.add_argument('--use-diffusion', type=int, default=1)
    parser.add_argument('--n-internal-steps', type=int, default=1)

    # initialization parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init-gt', type=int, default=0)

    # logging parameters
    parser.add_argument('--log-every', type=int, default=10)

    args = parser.parse_args()
    main(args)
