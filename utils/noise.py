import torch

def add_noise(images, noise_factor=0.2):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0., 1.)
