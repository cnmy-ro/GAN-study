import numpy as np
import pandas as pd
import torch


class MNISTDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, subset_name='test'):
		self.data_dir = data_dir
		self.subset_name = subset_name
		self.images, self.class_labels = load_mnist(data_dir, subset_name)
		self.images = self.images / 255.0

	def __len__(self):
		return self.images.shape[0]


	def __getitem__(self, idx):
		image = self.images[idx]
		image = torch.from_numpy(image).unsqueeze(dim=0)

		# label = self.class_labels[idx]
		# label = torch.from_numpy(label)

		sample = {'image': image}
		return sample



def load_mnist(data_dir, subset_name='test'):
	data = pd.read_csv(f"{data_dir}/mnist_{subset_name}.csv").values
	class_labels = data[:, 0]
	images = data[:, 1:].reshape(-1, 28, 28)
	return images, class_labels



if __name__ == '__main__':

	import matplotlib.pyplot as plt

	data_dir = "/home/chinmay/Datasets/MNIST"
	images, class_labels = load_mnist(data_dir)

	sample_idx = 0
	print("Label:", class_labels[sample_idx])
	plt.imshow(images[sample_idx])
	plt.show()


