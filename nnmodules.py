import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Generator1(nn.Module):
	def __init__(self, z_length):
		super().__init__()
		#self.z_length = z_length

		# Modules
		self.linear_1 = nn.Linear(z_length, 16)

		self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1)
		self.conv_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding=1)
		self.conv_3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), padding=1)
		self.conv_4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3,3), padding=1)
		self.conv_5 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), padding=1)


	def forward(self, z):
		x = F.relu(self.linear_1(z))

		x = x.view(-1, 1, 4, 4)
		x = F.relu(self.conv_1(x))

		x = F.interpolate(x, scale_factor=(2.0, 2.0), mode='bilinear')
		x = F.relu(self.conv_2(x))
		x = F.relu(self.conv_3(x))

		x = F.interpolate(x, size=(28, 28), mode='bilinear')
		x = F.relu(self.conv_4(x))
		x = F.sigmoid(self.conv_5(x))

		return x


class Generator2(nn.Module):
    def __init__(self,latent_dims):
        super(Generator2, self).__init__()

        self.img_shape = [1, 28, 28]

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dims, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator1(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1)
		self.maxpool_1 = nn.MaxPool2d(kernel_size=(2,2))

		self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=1)
		self.maxpool_2 = nn.MaxPool2d(kernel_size=(2,2))

		self.conv_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1)
		self.maxpool_3 = nn.MaxPool2d(kernel_size=(2,2))

		self.linear_1 = nn.Linear(32*3*3, 16)
		self.linear_2 = nn.Linear(16, 8)
		self.linear_3 = nn.Linear(8, 1)


	def forward(self, img):
		y = F.relu(self.maxpool_1(self.conv_1(img)))
		y = F.relu(self.maxpool_2(self.conv_2(y)))
		y = F.relu(self.maxpool_3(self.conv_3(y)))

		y = y.view(-1, y.shape[1]*y.shape[2]*y.shape[3])
		y = F.relu(self.linear_1(y))
		y = F.relu(self.linear_2(y))
		y = F.sigmoid(self.linear_3(y))

		return y


class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.img_shape = [1, 28, 28]

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity