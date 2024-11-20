import argparse
import torch
import os
import numpy as np
from scipy.linalg import svd, inv
import torch.nn.functional as F
import mrcfile
import pickle
from chroma import Protein
from chroma import Chroma
from chroma.layers.structure.rmsd import CrossRMSD

from src.plots import plot_metric, save_trajectory, plot_rmsd_ca_vs_completeness
from src.cif_utils import ma_cif_to_X
from src.fft import ifft_density
from src.pbe_solvers import RealisticSolver


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
    def multiply_covariance(dU, C): return backbone_network.noise_perturb.base_gaussian.multiply_covariance(dU, C)

    print("Initializing PBE solver")
    protein = Protein.from_CIF(args.cif, device='cuda')
    X_gt, C_gt, S_gt = protein.to_XCS(all_atom=False)  # we use X_gt to compute RMSD
    mask_gt = (C_gt == 1)[0]
    solver = RealisticSolver(
        args.mrc, S=S_gt, remove_oxt=args.remove_oxt, normalization='constant', resolution=args.resolution, unpad_len=args.unpad_len, outdir=args.outdir
    )
    delta = solver.origin
    X_gt -= delta  # align X_gt with the density
    density_gt = torch.tensor(solver.density)
    f_density_gt = ifft_density(density_gt).cuda()
    n_residues = X_gt.shape[1]

    print("Generating full density")
    X_gt_full, C_gt_full, S_gt_full = protein.to_XCS(all_atom=True)
    chi_gt, mask_chi = design_network.X_to_chi(X_gt_full, C_gt_full, S_gt_full)
    X_gt_full -= delta
    f_density = solver.get_full_f_density_per_batch(X_gt_full, args.n_voxels_per_batch)  # this density is unnormalized
    density = torch.abs(ifft_density(f_density)).float().cpu().numpy()
    
    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}_from_X_gt.mrc")
    with mrcfile.new(f"{args.outdir}/{args.outdir.split('/')[-1]}_from_X_gt.mrc", overwrite=True) as mrc:
        mrc.set_data(density)
    with mrcfile.open(f"{args.outdir}/{args.outdir.split('/')[-1]}_from_X_gt.mrc", mode='r+') as mrc:
        mrc.voxel_size = solver.voxel_size
        mrc.header.cella = solver.mrc_header.cella
        mrc.header.origin = solver.mrc_header.origin

    print("Precomputing preconditioning matrices")
    try:
        protein = Protein.from_CIF(args.ma_cif, device='cuda')
        X_ma, C_ma, _ = protein.to_XCS()
        mask_ma = C_ma[0].cpu().reshape(-1) > 0.5
    except ValueError:
        X_ma, mask_boundaries = ma_cif_to_X(args.ma_cif, X_gt.shape[1])
        X_ma = X_ma.cuda()
        mask_ma = mask_boundaries.reshape(-1) > 0.5
    delta = (X_gt[0, mask_ma * mask_gt.cpu()] - X_ma[0, mask_ma * mask_gt.cpu()]).mean(dim=(0, 1))
    X_ma += delta  # align the incomplete model on X_gt (which should be aligned with the density)
    Y = X_ma[:, mask_ma]

    def mR_fun(z):
        z = z.reshape(-1, n_residues, 4, 3)
        return multiply_R(z, C_gt.expand(z.shape[0], -1))[:, mask_ma].reshape(z.shape[0], -1).permute(1, 0)

    Z = torch.eye(n_residues * 12).cuda()
    mR_mat = mR_fun(Z).cpu().numpy()

    U, S, Vh = svd(mR_mat)
    Um1 = inv(U)
    Um1 = torch.tensor(Um1).float().cuda()[None].expand(args.population_size, -1, -1)
    S = torch.tensor(S).float().cuda()
    Vh = torch.tensor(Vh).float().cuda()[None].expand(args.population_size, -1, -1)

    print("Initializing backbone")
    C_gt = torch.abs(C_gt).expand(args.population_size, -1)
    S_gt = S_gt.expand(args.population_size, -1)
    chi_gt = chi_gt.expand(args.population_size, -1, -1)
    Y = Y.expand(args.population_size, -1, -1, -1)
    if args.init_gt:
        X = torch.clone(X_gt).expand(args.population_size, -1, -1, -1) + args.eps_init
        if args.std_dev_init > 1e-8:
            X += args.std_dev_init * torch.randn_like(X)
        Z = multiply_R_inverse(X, C_gt)
    else:
        Z = torch.randn(args.population_size, *X_gt.shape[1:]).float().cuda()
        X = multiply_R(Z, C_gt)
    V_m = torch.zeros_like(Z)
    V_d = torch.zeros_like(Z)
    V_s = torch.zeros_like(Z)
    V_c = torch.zeros_like(Z)
    
    def sample_chi(X, t=0):
        print("Sampling side chain angles")
        _X = F.pad(X, [0, 0, 0, 10])
        node_h, edge_h, edge_idx, mask_i, mask_ij = design_network.encode(_X, C_gt, t=t)
        permute_idx = design_network.traversal(_X, C_gt)
        _, chi_sample, _, logp_chi, _ = design_network.decoder_chi.decode(
            _X,
            C_gt,
            S_gt,
            node_h,
            edge_h,
            edge_idx,
            mask_i,
            mask_ij,
            permute_idx
        )
        return chi_sample
    if args.use_gt_chi:
        chi_sample = chi_gt
    else:
        chi_sample = torch.clone(sample_chi(X, t=1)).detach() if args.sample_chi_every > 0 else None

    def t_fn(epoch):
        if args.temporal_schedule == 'linear':
            return (-args.t + 0.001) * epoch / args.epochs + args.t
        elif args.temporal_schedule == 'sqrt':
            return (1.0 - 0.001) * (1. - np.sqrt(epoch / args.epochs)) + 0.001
        elif args.temporal_schedule == 'constant':
            return args.t
        else:
            raise NotImplementedError

    def lr_fn(epoch):
        return args.lr_density

    def resolution_fn(epoch):
        if epoch < args.activate_resolution_drop:
            return args.resolution_cutoff_start
        else:
            a = (args.resolution_cutoff_end - args.resolution_cutoff_start) / (args.epochs - args.activate_resolution_drop)
            b = args.resolution_cutoff_start - args.activate_resolution_drop * a
            return a * epoch + b

    def sampling_rate_fn(epoch):
        if args.sampling_rate_schedule == 'constant':
            return args.sampling_rate_start
        elif args.sampling_rate_schedule == 'linear':
            return args.sampling_rate_start + epoch * (args.sampling_rate_end - args.sampling_rate_start) / args.epochs
        elif args.sampling_rate_schedule == 'exp':
            return np.exp(np.log(args.sampling_rate_start) + epoch * (np.log(args.sampling_rate_end) - np.log(args.sampling_rate_start)) / args.epochs)
        else:
            raise NotImplementedError
    
    def density_error(d, d_gt, mode='mean'):
        if mode == 'sum':
            if args.normalize_detach:
                norm = torch.linalg.norm(d.reshape(d.shape[0], -1), dim=-1, keepdim=True).detach()
                norm_gt = torch.linalg.norm(d_gt).detach()
            else:
                norm = 4375.0  # hard-coded parameter -- has roughly the same role as the learning rate
                norm_gt = 1.
            neg_scalar_prod = (-torch.real(d.reshape(d.shape[0], -1) * d_gt.reshape(1, -1)) / (norm * norm_gt)).sum(-1)
            return neg_scalar_prod.sum(), neg_scalar_prod.detach()
        elif mode == 'mean':
            return (torch.abs(d - d_gt) ** 2).mean(dim=(1, 2, 3)).sum(), (torch.abs(d - d_gt) ** 2).mean(dim=(1, 2, 3)).detach()

    def get_gradient_Z_m(Z, t, epoch):
        if args.lr_model > 0. and (args.de_activate_model < 0 or epoch < args.de_activate_model):
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _Z = Z
                if args.preconditioning_model:
                    Um1Y = torch.bmm(Um1, Y.reshape(Y.shape[0], -1, 1)).reshape(Y.shape[0], -1)
                    Sm1Um1Y = Um1Y / S
                    loss_m = ((torch.bmm(Vh, _Z.reshape(Z.shape[0], -1, 1))[:, :Um1.shape[1], 0] - Sm1Um1Y) ** 2).sum()
                else:
                    loss_m = ((multiply_R(_Z, C_gt)[:, mask_ma] - Y) ** 2).sum()
                loss_m.backward()
                grad_Z_m = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_m = torch.zeros(*Z.shape).float().to(X.device)
            loss_m = torch.tensor([0.]).float().cuda()
        return grad_Z_m, loss_m

    def get_gradient_Z_d(Z, chi_sample, t, epoch):
        if args.lr_density > 0. and epoch >= args.activate_density:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _X = multiply_R(Z, C_gt)
                if args.sample_chi_every > 0 and epoch + 1 % args.sample_chi_every == 0:
                    if args.use_gt_chi:
                        chi_sample = chi_gt
                    else:
                        chi_sample = sample_chi(_X, t)
                _X_full, _ = design_network.chi_to_X(_X, C_gt, S_gt, chi_sample)
                f_density_sampled, f_density_sampled_gt = solver.get_one_batch_f_density(
                    _X_full, f_density_gt, resolution_fn(epoch), sampling_rate=sampling_rate_fn(epoch)
                )
                loss_d, loss_d_per_sample = density_error(f_density_sampled, f_density_sampled_gt, mode='sum')
                loss_d.backward()
                grad_Z_d = Z.grad * (resolution_fn(epoch) / args.resolution_cutoff_end)**3 / sampling_rate_fn(epoch)  # scale by epoch-dependent factors 
            Z.requires_grad_(False)
        else:
            grad_Z_d = torch.zeros(*Z.shape).float().to(X.device)
            loss_d = torch.tensor([0.]).float().cuda()
            loss_d_per_sample = torch.zeros(Z.shape[0]).float().cuda()
        return grad_Z_d, loss_d, chi_sample, loss_d_per_sample

    def get_gradient_Z_s(Z, t, epoch):
        if args.lr_sequence > 0. and epoch >= args.activate_sequence:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _X = multiply_R(Z, C_gt)
                _X_input = F.pad(_X, [0, 0, 0, 10])
                out = design_network(_X_input, C_gt, S_gt, t.cuda())
                logp_S = out["logp_S"]
                loss_s = -logp_S.sum()
                loss_s.backward()
                grad_Z_s = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_s = torch.zeros(*Z.shape).float().to(X.device)
            loss_s = torch.tensor([0.]).float().cuda()
        return grad_Z_s, loss_s

    def get_gradient_Z_c(Z):
        if args.lr_inter_ca > 0.:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _X = multiply_R(Z, C_gt)
                distances = torch.linalg.norm(_X[:, 1:, 1] - _X[:, :-1, 1])
                loss_c = ((distances - 3.8) ** 2).sum()
                loss_c.backward()
                grad_Z_c = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_c = torch.zeros(*Z.shape).float().to(X.device)
            loss_c = torch.tensor([0.]).float().cuda()
        return grad_Z_c, loss_c
    
    trajectory = [torch.clone(X_gt[:, mask_gt]).detach().cpu().numpy(),
                  torch.clone(Y[:1]).detach().cpu().numpy(),
                  (torch.clone(X).detach().cpu().numpy(), 'initial state')]
    
    metrics = {'epoch': [], 'rmsd': [], 't': [], 'loss_m': [], 'loss_d': [], 'rmsd_ca': [],
               'resolution': [], 'loss_s': [], 'lr_density': [], 'loss_d_per_sample': [], 'sampling_rate': [],
               'loss_c': []}

    print("--- Optimization starts now ---")
    for epoch in range(args.epochs):
        t = torch.tensor(t_fn(epoch)).float().cuda()

        if args.use_diffusion:
            with torch.no_grad():
                X0 = backbone_network.denoise(X.detach(), C_gt, t)
        else:
            X0 = X
        Z0 = multiply_R_inverse(X0, C_gt)

        grad_Z_m, loss_m = get_gradient_Z_m(Z0, t, epoch)
        V_m = args.rho_model * V_m + args.lr_model * grad_Z_m

        grad_Z_d, loss_d, chi_sample, loss_d_per_sample = get_gradient_Z_d(Z0, chi_sample, t, epoch)
        chi_sample = torch.clone(chi_sample).detach() if chi_sample is not None else None
        V_d = args.rho_density * V_d + lr_fn(epoch) * grad_Z_d

        grad_Z_s, loss_s = get_gradient_Z_s(Z0, t, epoch)
        V_s = args.rho_sequence * V_s + args.lr_sequence * grad_Z_s

        grad_Z_c, loss_c = get_gradient_Z_c(Z0)
        V_c = args.rho_inter_ca * V_c + args.lr_inter_ca * grad_Z_c

        Z0 = Z0 - V_m - V_d - V_s - V_c

        # replicate models with lowest density error
        if args.select_best_every > 0 and epoch >= args.activate_replication and epoch % args.select_best_every == 0:
            assert Z0.shape[0] % args.replication_factor == 0, "The population size must be an integer multiple of the replication factor"
            _, indices = torch.topk(loss_d_per_sample, Z0.shape[0] // args.replication_factor, largest=False)
            Z0 = torch.clone(Z0[indices][:, None].expand(-1, args.replication_factor, -1, -1, -1)).reshape(-1, *Z0.shape[1:])

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
                    torch.clone(X[i, mask_gt]).cpu().reshape(1, -1, 3),
                    torch.clone(X_gt[0, mask_gt]).cpu().reshape(1, -1, 3),
                    compute_alignment=True
                )
                rmsd_ca, _ = CrossRMSD().pairedRMSD(
                    torch.clone(X[i, mask_gt, 1, :]).cpu().reshape(1, -1, 3),
                    torch.clone(X_gt[0, mask_gt, 1, :]).cpu().reshape(1, -1, 3),
                    compute_alignment=True
                )
                rmsds.append(rmsd.item())
                rmsds_cas.append(rmsd_ca.item())
            idx_best = np.argmin(rmsds)
            rmsd_best = rmsds[idx_best]
            rmsd_ca_best = rmsds_cas[idx_best]

            rmsd, X_aligned = CrossRMSD().pairedRMSD(
                torch.clone(X[idx_best, mask_gt]).cpu().reshape(1, -1, 3),
                torch.clone(X_gt[0, mask_gt]).cpu().reshape(1, -1, 3),
                compute_alignment=True
            )
            X_aligned = X_aligned.reshape(1, -1, 4, 3)

            metrics['epoch'].append(epoch)
            metrics['t'].append(t.item())
            metrics['loss_m'].append(loss_m.item())
            metrics['loss_d'].append(loss_d.item())
            metrics['loss_d_per_sample'].append(loss_d_per_sample)
            metrics['loss_s'].append(loss_s.item())
            metrics['loss_c'].append(loss_c.item())
            metrics['resolution'].append(resolution_fn(epoch))
            metrics['sampling_rate'].append(sampling_rate_fn(epoch))
            metrics['lr_density'].append(lr_fn(epoch))
            metrics['rmsd'].append(rmsds)
            metrics['rmsd_ca'].append(rmsds_cas)
            trajectory.append((torch.clone(X[idx_best][None]).detach().cpu().numpy(), 'x-update'))
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss Model: {loss_m.item():.4e}, Loss Density: {loss_d.item():.4e}, RMSD: {rmsd_best:.2e}, RMSD CA: {rmsd_ca_best:.2e}")

    C_gt = C_gt[0:1]
    S_gt = S_gt[0:1]
    X_full, _ = design_network.chi_to_X(X[idx_best][None], C_gt, S_gt, chi_sample)

    print(f"Saving {args.outdir}/metrics.pkl")
    with open(f"{args.outdir}/metrics.pkl", 'wb') as file:
        pickle.dump(metrics, file)

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}.mrc")
    f_density = solver.get_full_f_density_per_batch(X_full, args.n_voxels_per_batch)
    density = torch.real(ifft_density(f_density))
    density = density.cpu().numpy()
    with mrcfile.new(f"{args.outdir}/{args.outdir.split('/')[-1]}.mrc", overwrite=True) as mrc:
        mrc.set_data(density)
    with mrcfile.open(f"{args.outdir}/{args.outdir.split('/')[-1]}.mrc", mode='r+') as mrc:
        mrc.voxel_size = solver.voxel_size
        mrc.header.cella = solver.mrc_header.cella
        mrc.header.origin = solver.mrc_header.origin

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}.pdb")
    X_full += solver.origin
    protein_out = Protein.from_XCS(X_full, C_gt, S_gt)
    protein_out.to_PDB(f"{args.outdir}/{args.outdir.split('/')[-1]}.pdb")

    for key in metrics.keys():
        if key != 'epoch' and key != 'loss_d_per_sample':
            print(f"Saving {args.outdir}/{key}.png")
            plot_metric(metrics, key, f"{args.outdir}/{key}.png")

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}.mp4")
    save_trajectory(trajectory, f"{args.outdir}/{args.outdir.split('/')[-1]}.mp4")

    print(f"Saving {args.outdir}/rmsd_ca_vs_completeness.png")
    plot_rmsd_ca_vs_completeness(X_gt, X_ma, X[idx_best][None], mask_gt.cpu(), mask_ma, f"{args.outdir}/rmsd_ca_vs_completeness.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--outdir', type=str, required=True, help="Path to output directory.")
    parser.add_argument('--mrc', type=str, required=True, help="Path to density map in the MRC file.")
    parser.add_argument('--ma-cif', type=str, required=True, help="Path to incomplete model (e.g., ModelAngelo output), in the CIF format.")
    parser.add_argument('--cif', type=str, required=True, help="Path to deposited CIF file.")

    # I/O parameters
    parser.add_argument('--remove-oxt', type=int, default=1, help="Flag to ignore terminal oxygen.")
    parser.add_argument('--weights-backbone', type=str, default=None, help="Path to Chroma weights (backbone).")
    parser.add_argument('--weights-design', type=str, default=None, help="Path to Chroma weights (design).")
    parser.add_argument('--resolution', type=float, default=2.0, help="Resolution of the density map")
    parser.add_argument('--unpad-len', type=int, default=0, help="Number of (empty) voxels to remove on each side of the input voxel grid, to speed up computation.")

    # optimization parameters
    parser.add_argument('--epochs', type=int, default=4000, help="Number of epochs.")
    parser.add_argument('--population-size', type=int, default=16, help="Number of atomic models to simultaneously optimize.")
    parser.add_argument('--lr-model', type=float, default=1e-2, help="Learning rate for the model loss.")
    parser.add_argument('--rho-model', type=float, default=0.9, help="Momentum for the model loss.")
    parser.add_argument('--lr-density', type=float, default=1e-2, help="Learning rate for the density loss.")
    parser.add_argument('--rho-density', type=float, default=0.9, help="Momentum for the density loss.")
    parser.add_argument('--lr-sequence', type=float, default=1e-5, help="Learning rate for the sequence loss.")
    parser.add_argument('--rho-sequence', type=float, default=0.9, help="Momentum for the sequence loss.")
    parser.add_argument('--lr-inter-ca', type=float, default=0.0, help="Learning rate for the inter-CA loss.")
    parser.add_argument('--rho-inter-ca', type=float, default=0.9, help="Momentum for the inter-CA loss.")
    parser.add_argument('--preconditioning-model', type=int, default=1, help="Flag to use preconditioning on the model loss.")
    parser.add_argument('--normalize-detach', type=int, default=0, help="Normalize Fourier map before computing the density loss.")
    parser.add_argument('--de-activate-model', type=int, default=-1, help="Number of epochs before de-activating the model loss (-1 to always activate).")
    parser.add_argument('--activate-density', type=int, default=0, help="Number of epochs before activating density loss.")
    parser.add_argument('--activate-sequence', type=int, default=3000, help="Number of epochs before activating sequence loss.")

    # diffusion parameters
    parser.add_argument('--use-diffusion', type=int, default=1, help="Flag to use the diffusion model.")
    parser.add_argument('--temporal-schedule', type=str, default='sqrt', choices=['sqrt', 'linear', 'constant'], help="Type of temporal schedule.")
    parser.add_argument('--t', type=float, default=1.0, help="Initial diffusion time (between 0 and 1).")

    # resolution
    parser.add_argument('--resolution-cutoff-start', type=float, default=1.5, help="Initial resolution to compute the (Fourier) density maps up to.")
    parser.add_argument('--resolution-cutoff-end', type=float, default=1.5, help="Final resolution to compute the (Fourier) density maps up to.")
    parser.add_argument('--activate-resolution-drop', type=int, default=0, help="Number of epochs before changing the resolution.")

    # random sampling
    parser.add_argument('--sampling-rate-schedule', type=str, default='constant', choices=['constant', 'linear', 'exp'], help='Type of schedule for the sampling rate.')
    parser.add_argument('--sampling-rate-start', type=float, default=0.1, help='Initial sampling rate.')
    parser.add_argument('--sampling-rate-end', type=float, default=1.0, help='Final sampling rate.')

    # side-chain parameters
    parser.add_argument('--sample-chi-every', type=int, default=100, help="Frequency (in epochs) of side-chain sampling.")
    parser.add_argument('--use-gt-chi', type=int, default=0, help="Flag to use ground truth side-chain angles, for debugging purposes.")
    
    # genetic parameters
    parser.add_argument('--replication-factor', type=int, default=2, help='Number of replications at each selection step.')
    parser.add_argument('--activate-replication', type=int, default=1, help="Number of epochs to wait before activating the selection/replication.")
    parser.add_argument('--select-best-every', type=int, default=500, help='Frequency (in epochs) of selection/replication (-1 to de-activate).')

    # initialization parameters
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--init-gt', type=int, default=0, help="Flag to initialize the model from the deposited CIF, for debugging purposes.")
    parser.add_argument('--std-dev-init', type=float, default=0.0, help="Intensity of Gaussian random noise added on ground truth.")
    parser.add_argument('--eps-init', type=float, default=0.0, help="Size of initial deviation to ground truth in the direction (1, 1, 1).")

    # other parameters
    parser.add_argument('--n-voxels-per-batch', type=int, default=1024, help="Number of voxels per batch to avoid OOM when computing full density maps.")

    # logging parameters
    parser.add_argument('--log-every', type=int, default=10, help="Frequency (in epochs) for logging.")

    args = parser.parse_args()
    main(args)
