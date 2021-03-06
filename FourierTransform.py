#import cupy as np
import numpy as np


def ft_2d_naive(image):
	# This method cannot be executed with my laptop since it requires more than 4 GiB of memory.
	image_x, image_y = image.shape[:2]
	x_sum = np.arange(image_x)
	y_sum = np.arange(image_y)
	ft_matrix = np.zeros((image_x, image_y),np.complex64)
	
	for u in x_sum:
		exponent = -2 * np.pi * (0+1j) * ((u * x_sum / image_x)[:, np.newaxis, np.newaxis] + (y_sum[:,np.newaxis] * y_sum/image_y))
		ft_matrix[u] = (image[:,np.newaxis] * np.exp(exponent)).sum(axis = 2).sum(axis = 0)
		print("{0} / {1}".format(u + 1, image_x), end = "\r")
	return(ft_matrix)
	
def ft_2d_2loops(image):
	"""Naive 2d Fourier transform implementation with double for loop.
	This method is left in here for progress record purpose"""
	image_x, image_y = image.shape[:2]
	x_sum = np.arange(image_x) - (image_x // 2)
	y_sum = np.arange(image_y) - (image_y // 2)
	ft_matrix = np.zeros((image_x, image_y),np.complex64)
	
	for u in x_sum:
		for v in y_sum:
			exponent = -2 * np.pi * (0+1j) * ((u * x_sum / image_x)[:, np.newaxis] + (v * y_sum/image_y))
			ft_matrix[u,v] = (image*np.exp(exponent)).sum()
	return(ft_matrix)

def inverse_ft2(ft_image):
	image_x, image_y = ft_image.shape[:2]
	x_sum = np.arange(image_x)
	y_sum = np.arange(image_y)
	ft_matrix = np.zeros((image_x, image_y),np.complex64)
	
	for u in x_sum:
		exponent = 2 * np.pi * (0+1j) * ((u * x_sum / image_x)[:, np.newaxis, np.newaxis] + (y_sum[:,np.newaxis] * y_sum/image_y))
		ft_matrix[u] = (ft_image[:,np.newaxis] * np.exp(exponent)).sum(axis = 2).sum(axis = 0)
		print("{0} / {1}".format(u + 1, image_x), end = "\r")
	return(ft_matrix)

def inverse_ft2_2loops(ft_image):
	image_x, image_y = ft_image.shape[:2]
	x_sum = np.arange(image_x) - (image_x // 2)
	y_sum = np.arange(image_y) - (image_y // 2)
	ft_matrix = np.zeros((image_x, image_y),np.complex64)
	
	for u in x_sum:
		for v in y_sum:
			exponent = 2 * np.pi * (0+1j) * ((u * x_sum / image_x)[:, np.newaxis] + (v * y_sum/image_y))
			ft_matrix[u,v] = (ft_image*np.exp(exponent)).sum()
	return(ft_matrix)
	
def custom_fft2(image):
	pass
	
def display_fourier(fourier):
	plt.imshow(np.log(np.abs(np.fft.fftshift(fourier))),'gray')
	