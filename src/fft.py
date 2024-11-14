import torch


def fft_density(d):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(d, dim=(-3,-2,-1)), dim=(-3,-2,-1)), dim=(-3,-2,-1))


def ifft_density(fd):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(fd, dim=(-3,-2,-1)), dim=(-3,-2,-1)), dim=(-3,-2,-1))