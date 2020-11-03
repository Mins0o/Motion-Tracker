#import cupy as np
import numpy as np
import matplotlib.pyplot as plt
import FourierTransform

import cv2

def template_matching(template,target):
	"""I wrote the methods below purely for my amusement and learning.
	That's why I went extra miles with unoptimized approach and my own crude algorithms.
	This method will utilize optimized methods from external library and will be used for practical performance."""
	target = pad_image(target)
	target = (target*256) - 0.5
	target = target.astype("int32")
	template = (template*256) - 0.5
	template = template.astype("int32")
	match = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
	print(match)
	return(match)

def naive_convolve(target, kernel, verbose = True, pad = False):
	"""
	target: (nparray) target image.
	kernel: (nparray) The kernel (filter) you want to apply to the image.
			The dimensions should be odd numbers.
	verbose: (Boolean) If True, plot of the result is shown.
	pad : (Boolean) If True, the target image is padded with zeros before convolving. Default is False.
	"""
	
	# The number of channels is not handled here because grayscale images-
	# - cause index error. It is handled in the if statements below.
	size_x, size_y = target.shape[:2]
	kernel_x, kernel_y = kernel.shape
	# kernel shape validity check
	if((kernel_x//2) * (kernel_y//2)) == 0:
		print("Kernel sizes should be odd numbers in both axises")
		return(target)
	
	if pad:
		target = pad_image(target,(kernel_x, kernel_y))
	
	convolved_image = np.zeros((size_x - kernel_x + 1,size_y - kernel_y + 1, 3))
	
	
	if not target.dtype == 'float':
		target.dtype = 'float'
		target = (target + 0.5) / 256
		if target.max() > 1:
			print("Image's max value exceeds the limit")
		
	# One channel
	if len(target.shape) == 2 or target.shape[2] == 1:
		convolved_image = naive_single_channel(target, kernel)
	
	elif target.shape[2] == 3:
		# separate channels
		channel1 = target[:,:,0]
		channel2 = target[:,:,1]
		channel3 = target[:,:,2]
		convolved1 = naive_single_channel(channel1, kernel)
		convolved2 = naive_single_channel(channel2, kernel)
		convolved3 = naive_single_channel(channel3, kernel)
		channels = (convolved1, convolved2, convolved3)
		convolved_image = np.stack(channels, axis = 2)
	else:
		print("We are not ready to process this type of image yet")
		return convolved_image
		
	# This should be done to absolute value if you want to properly show the image
	_temp_zero = convolved_image < 0
	_temp_one = convolved_image > 1
	convolved_image[_temp_zero] = 0
	convolved_image[_temp_one] = 1
	
	if verbose:
		#plt.imshow(convolved_image.get(), "gray")
		plt.imshow(convolved_image, "gray")
		plt.show()
	return convolved_image
	
def pad_image(target, kernel_size = (3, 3), value = "zero"):
	"""
	target: nparray of the image
	kernel_size: The size of the kernel being used.
				tuple with two integers (both numbers are odd numbers).
	value: The pattern value to fill in the padding
			-zero: default. Fills in the padding with zero
			-symmetric: (sym) Fills in the padding with mirrored image
			-tile: Fills in the padding as if the image is tiled (repeated).
	"""
	target_x, target_y = np.shape(target)[:2]
	kernel_x, kernel_y = kernel_size
	if len(target.shape) == 2 or target.shape[2] == 1:
		new_image = np.zeros((target_x + kernel_x - 1, target_y + kernel_y - 1))
	elif target.shape[2] == 3:
		new_image = np.zeros((target_x + kernel_x - 1, target_y + kernel_y - 1, 3))
	new_image[kernel_x //2:kernel_x // 2 + target_x, kernel_y // 2:kernel_y // 2 + target_y] = target
	if value.lower() == "sym" or value.lower == "symmetric":
		pass
	elif value.lower() == "tile":
		pass
	return(new_image)
	
def naive_single_channel(target, kernel):
	"""
	This method just simply convolves the images as if there is no padding. 
	The padding could be added prior to calling this method.
	"""
	# Get the size information
	size_x, size_y = target.shape[:2]
	kernel_x, kernel_y = kernel.shape[:2]
	
	# Create an empty convolved image template
	# Notice the result size is smaller than the input image.
	convolved_image = np.zeros((size_x - kernel_x + 1, size_y - kernel_y + 1))
	
	# Loop throught the target image with the kernel
	for x in range(kernel_x //2, size_x - kernel_x //2):
		for y in range(kernel_y // 2, size_y - kernel_y // 2):
			# Cut out the target image in to the same size as the kernel, with (x, y) in the middle
			# From (x - kernel_x//2) to (x + kernel_x//2 + 1)
			# From (y - kernel_y//2) to (y + kernel_y//2 + 1)
			target_window = target[x - kernel_x // 2 : x + kernel_x // 2 + 1, y - kernel_y // 2 : y + kernel_y // 2 + 1]
			# Convolve
			# Target image (x,y) => convolved image (x - kernel//2, y -kernel//2)
			convolved_image[x - kernel_x // 2, y - kernel_y //2] = np.sum(np.multiply(target_window, np.rot90(kernel,2)))
	
	return convolved_image

# Just convolving/correlating a template for matching does not work as matching
# The kind of correlation that works is called Zero Normalized Cross Correlation (ZNCC)
def naive_matching(target, template, verbose = True):
	# Get the size information
	size_y, size_x = target.shape[:2]
	template_y, template_x = template.shape[:2]
	
	# Create an empty convolved image template
	match_y = size_y - template_y + 1
	match_x = size_x - template_x + 1
	match_result = np.zeros((match_y, match_x))
	
	# Normalizing by subtracting the mean value
	target = target - np.mean(target)
	template = template - np.mean(template)
	
	
	# Loop throught the target image with the template
	for x in range(template_x //2, size_x - template_x //2):
		for y in range(template_y // 2, size_y - template_y // 2):
			# Cut out the target image in to the same size as the template, with (x, y) in the middle
			# From (x - template_x//2) to (x + template_x//2 + 1)
			# From (y - template_y//2) to (y + template_y//2 + 1)
			target_window = target[y - template_y // 2 : y + template_y // 2 + 1, x - template_x // 2 : x + template_x // 2 + 1]
			# Convolve
			# Target image (x,y) => convolved image (x - template//2, y -template//2)
			match_result[y - template_y //2, x - template_x // 2] = np.sum(np.multiply(target_window, template))
	
	match_result = match_result - np.min(match_result)
	match_result = match_result / np.max(match_result)
	if verbose:
		plt.imshow(match_result,'gray')
		plt.show()
	match_point = np.argmax(match_result)
	# These coordinates are for the ORIGINAL IMAGE not the result image!
	x_matchpoint = match_point % match_x + template_x//2
	y_matchpoint = match_point // match_x + template_y//2
	return((x_matchpoint, y_matchpoint),match_result)

def convolution_ft(image,kernel):
	"""This method takes up too much time"""
	image_x, image_y = image.shape[:2]
	kernel_x, kernel_y = image.shape[:2]
	
	if not image.dtype == 'float':
		image.dtype = 'float'
		image = (image + 0.5) / 256
		if image.max() > 1:
			print("Image's max value exceeds the limit")
	
	if kernel_x > image_x or kernel_y > image_y:
		print("The target image must have bigger dimension than the kernel")
		return image
		
	kernel = extend_kernel(kernel, (image_x, image_y))
	ft_kernel = FourierTransform.ft_2d_2loops(kernel)
	
	# One channel
	if len(image.shape) == 2 or target.shape[2] == 1:
		ft_image = FourierTransform.ft_2d_2loops(image)
		convolved_ft = ft_image * ft_kernel
		convolved_image = FourierTransform.inverse_ft2_2loops(convolved_ft)
	elif target.shape[2] == 3:
		# separate channels
		channel1 = target[:,:,0]
		channel2 = target[:,:,1]
		channel3 = target[:,:,2]
		ft_ch1 = FourierTransform.ft_2d_2loops(channel1)
		ft_ch2 = FourierTransform.ft_2d_2loops(channel2)
		ft_ch3 = FourierTransform.ft_2d_2loops(channel3)
		convolved1 = np.abs(FourierTransform.inverse_ft2_2loops(ft_ch1 * ft_kernel))
		convolved2 = np.abs(FourierTransform.inverse_ft2_2loops(ft_ch2 * ft_kernel))
		convolved3 = np.abs(FourierTransform.inverse_ft2_2loops(ft_ch3 * ft_kernel))
		channels = (convolved1, convolved2, convolved3)
		convolved_image = np.stack(channels, axis = 2)
	else:
		print("We are not ready to process this type of image yet")
		return convolved_image
	
	return convolved_image

def extend_kernel(kernel, target_size = (100,100)):
	target_x, target_y = target_size
	kernel_x, kernel_y = kernel.shape
	ext_kernel = np.zeros((target_x, target_y))
	ext_kernel[:kernel_x, :kernel_y] = kernel
	return ext_kernel
	

if __name__ == "__main__":
	# Images
	minsoo = plt.imread("./Img/Minsoo.jpg")/255
	hailey = plt.imread("./Img/Hailey.jpg")/255
	poodle = plt.imread("./Img/Poodle.jpg")/255
	ear = plt.imread("./Img/Ear.jpg")/255
	
	# GrayScale
	g_minsoo = np.average(minsoo, axis = 2)
	g_hailey = np.average(hailey, axis = 2)
	g_poodle = np.average(poodle, axis = 2)
	g_ear = np.average(ear, axis = 2)
	
	#kernels
	sobel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
	sobel2 = np.array([[-2, -2, -2],[-2, 16, -2],[-2, -2, -2]])
	
	translation = np.zeros((101,101))
	translation[0,0] = 1
	
	
	#naive_convolve(g_minsoo, sobel)
	#naive_convolve(g_minsoo, sobel2)
	#naive_convolve(minsoo,translation,True,True)
	#coord, match_img = naive_matching(g_minsoo, g_ear, False)
	match_img = template_matching(g_ear, g_minsoo)
	#coord, match_img = naive_matching(g_minsoo, translation, False)
	print(coord)
	plt.subplot(131)
	plt.imshow(g_minsoo,'gray')
	plt.subplot(132)
	plt.imshow(g_ear,'gray')
	plt.subplot(133)
	plt.imshow(match_img,'gray')
	plt.show()
	
	