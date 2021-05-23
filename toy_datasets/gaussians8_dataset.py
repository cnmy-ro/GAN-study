import numpy as np
from torch.utils.data import Dataset


MEANS = np.array([
	              [0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5],
	              [0.3, 0.3], [0.3, 0.7], [0.7, 0.3], [0.7, 0.7]
	             ])

STDDEV = np.array([0.001, 0.001])


class Gaussians8Dataset(Dataset):
	def __init__(self, n_samples=1600):
		super().__init__()
		self.n_samples = n_samples
		self.data, self.labels = generate_data(n_samples)

	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		sample = {'point': self.data[idx],
		          'label': self.data_label[idx]}
		return sample


def generate_data(n_samples):
	data = np.zeros((n_samples, 2))
	data_labels = np.zeros((n_samples, 1))

	for i in range(8):
		samples = np.random.multivariate_normal(mean=MEANS[i], cov=np.diag(STDDEV), size=n_samples//8)
		data[i*samples.shape[0]:(i+1)*samples.shape[0], :] = samples
		data_labels[i*samples.shape[0]:(i+1)*samples.shape[0]] = i

	return data, data_labels


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	data, data_labels = generate_data(1600)
	for i in range(8):
		plt.scatter(data[i*200:(i+1)*200, 0], data[i*200:(i+1)*200, 1], alpha=0.5, label=i)
	plt.legend()
	plt.show()