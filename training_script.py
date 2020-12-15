import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

#import sys
#sys.path.append("/gdrive/MyDrive/work_space/GAN_stuff")

from mnist_dataset import MNISTDataset
from nnmodules import Generator2, Discriminator2
import utils



DATA_DIR = "/home/chinmay/Datasets/MNIST"
DEVICE = "cpu"

Z_LENGTH = 8

N_EPOCHS = 100
BATCH_SIZE = 2
CHECKPOINT_DIR = "./"


def main():
	train_dataset = MNISTDataset(DATA_DIR, subset_name='train')
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

	generator = Generator2(Z_LENGTH).to(DEVICE)
	discriminator = Discriminator2().to(DEVICE)

	gen_optimizer = SGD(generator.parameters(), lr=0.0003, momentum=0.9)
	discrim_optimizer = SGD(discriminator.parameters(), lr=0.0003, momentum=0.9)
	criterion = nn.BCELoss(reduction='mean')



	# Training loop
	for epoch in range(1, N_EPOCHS):

	  print("Epoch:", epoch)
	  epoch_discrim_loss = 0
	  epoch_gen_loss = 0

	  for batch in train_loader:

	    # Discriminator phase --
	    real_images = batch['image'].to(DEVICE).float()
	    ones = torch.ones(real_images.shape[0], 1, device=DEVICE)

	    noise_batch = utils.get_noise_vector(z_size=(BATCH_SIZE, Z_LENGTH)).to(DEVICE)
	    with torch.no_grad():
	      fake_images = generator(noise_batch)
	    zeros = torch.zeros(fake_images.shape[0], 1, device=DEVICE)

	    image_batch = torch.cat((real_images, fake_images), dim=0)
	    labels_batch = torch.cat((ones, zeros), dim=0)
	    idxs = np.arange(image_batch.shape[0])
	    np.random.shuffle(idxs)
	    image_batch = image_batch[idxs]
	    labels_batch = labels_batch[idxs]

	    # Forward pass and update D
	    discrim_optimizer.zero_grad()
	    reals_loss = criterion(discriminator(real_images), ones)
	    fakes_loss = criterion(discriminator(fake_images), zeros)
	    discrim_loss = (reals_loss + fakes_loss) / 2
	    discrim_loss.backward(retain_graph=True)
	    discrim_optimizer.step()


	    # Generator phase --
	    gen_optimizer.zero_grad()
	    noise_batch = utils.get_noise_vector(z_size=(BATCH_SIZE, Z_LENGTH)).to(DEVICE)
	    fake_images = generator(noise_batch)
	    discrim_pred = discriminator(fake_images)
	    ones = torch.ones(BATCH_SIZE, 1, device=DEVICE)

	    # Update G
	    gen_loss = criterion(discrim_pred, ones)
	    gen_loss.backward()
	    gen_optimizer.step()

	    epoch_discrim_loss += discrim_loss.item()
	    epoch_gen_loss += gen_loss.item()


	  epoch_discrim_loss /= len(train_loader)
	  epoch_gen_loss /= len(train_loader)
	  print("D loss:", epoch_discrim_loss, "| G loss:", epoch_gen_loss)

	  if epoch % 1 == 0:
	    #sample_image = fake_images[0,0,:,:].cpu().detach().numpy()
	    grid = torchvision.utils.make_grid(fake_images.cpu().detach())
	    plt.imshow(grid.permute(1,2,0))
	    plt.show()


if __name__ == '__main__':
	main()