import numpy as np


MEANS = np.array([
	              [7.5, 7.5],
	              [7.5, 12.5],
	              [12.5, 7.5],
	              [12.5, 12.5]
	             ])

STDDEV = np.array([1, 1])



def generate_data(n_samples=1600):
	data2d = np.zeros((n_samples, 2))
	data_labels = np.zeros((n_samples, 1))

	for i in range(4):
		samples = np.random.multivariate_normal(mean=MEANS[i], cov=np.diag(STDDEV), size=n_samples//4)
		data2d[i*samples.shape[0]:(i+1)*samples.shape[0], :] = samples
		data_labels[i*samples.shape[0]:(i+1)*samples.shape[0]] = i

	data3d = np.zeros((n_samples, 3))
	xs, ys = data2d[:, 0], data2d[:, 1]
	data3d[:, 0] = xs * np.cos(xs)
	data3d[:, 1] = ys
	data3d[:, 2] = xs * np.sin(xs)

	return data3d, data_labels



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	data, data_labels = generate_data(1600)

	fig, (xy, yz, zx)  = plt.subplots(1, 3, figsize=(15, 5))
	for i in range(4):
		xy.scatter(data[i*400:(i+1)*400, 2], data[i*400:(i+1)*400, 1], alpha=0.25, label=i)
		yz.scatter(data[i*400:(i+1)*400, 1], data[i*400:(i+1)*400, 0], alpha=0.25, label=i)
		zx.scatter(data[i*400:(i+1)*400, 0], data[i*400:(i+1)*400, 2], alpha=0.25, label=i)

	xy.set_title("x-y")
	yz.set_title("y-z")
	zx.set_title("z-x")

	fig.tight_layout()
	plt.legend()
	plt.show()