import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms


class MNISTDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, subset_name='test'):
		self.data_dir = data_dir
		self.subset_name = subset_name
		self.images, self.class_labels = load_mnist(data_dir, subset_name)
		self.images = (self.images / 255.0) * 2 - 1 # Scale from [0,255] to [-1,1]
		self.transform = transforms.Compose(
			                                [transforms.ToPILImage(),
			                                 transforms.Resize([32,32]),
			                                 transforms.ToTensor()
			                                ]
			                               )
	def __len__(self):
		return self.images.shape[0]


	def __getitem__(self, idx):
		image = self.images[idx]
		image = self.transform(image)

		class_label = self.class_labels[idx]
		class_label = torch.from_numpy(class_label)

		sample = {'image': image, 'class-label': class_label}
		return sample



def load_mnist(data_dir, subset_name='test'):
	data = pd.read_csv(f"{data_dir}/mnist_{subset_name}.csv").values
	class_labels = data[:, 0:1].astype(np.int8)
	images = data[:, 1:].reshape(-1, 28, 28).astype(np.float32)
	return images, class_labels



if __name__ == '__main__':

	import matplotlib.pyplot as plt

	data_dir = "/home/chinmay/Datasets/MNIST"

	dataset = MNISTDataset(data_dir, subset_name='test')
	sample = dataset[0]
	print(sample['image'].shape)
	print(sample['class-label'].shape)