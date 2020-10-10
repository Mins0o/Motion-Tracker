import numpy as np


def ft_2d_naive(image):
	image_x, image_y = image.shape[:2]
	x_sum = np.arange(-image_x // 2, image_x // 2)[:,np.newaxis]
	x_exponent = (x_sum[:,np.newaxis] * x_sum) / image_x
	y_sum = np.arange(-image_y // 2, image_y // 2)[:,np.newaxis]
	y_exponent = (y_sum[:,np.newaxis] * y_sum) / image_y
	exp_matrix = np.exp(-2 * np.pi * (0+1j) * (x_exponent[:,:,np.newaxis,np.newaxis] + y_exponent))
	return(exp_matrix)