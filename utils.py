
import torch


def get_noise_vector(z_size):
	return torch.randn(z_size)