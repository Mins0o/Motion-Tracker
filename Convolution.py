import numpy as np
import matplotlib.pyplot as plt

def naive_convolve(target, kernel, verbose = True):
	"""
	1. Kernel should have odd number of pixels for horizontal and vertical dimensions. This is because we need to know the center point of where the kernel is being convolved.
	2. The act of convolving is done by np.sum(n.multiply(<Image>, <Kernel>)). (Yes, the image and kernel are in np.array type.)
	3. Image array should be float ranging between 0 to 1.
	4. Check the dtype and normalize to 0~1.
	"""
	size_x, size_y = target.shape[:2]
	kernel_x, kernel_y = kernel.shape[:2]
	# target shape validity check
	
	
	convolved_image = np.zeros((size_x - kernel_x + 1,size_y - kernel_y + 1, 3))
	
	
	if target.dtype == 'float':
		pass
	
	# one channel
	if len(target.shape) == 2 or target.shape[2] == 1:
		print("Single Channel")
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
		
	if verbose:
		plt.imshow(convolved_image, "gray", vmin = 0, vmax = 1)
		plt.show()
	return convolved_image
	
def pad_image(target, kernel_size = (3, 3)):
	"""
	The kernel_size parameter is a tuple with two integers.
	"""
	pass
	
def naive_single_channel(target, kernel):
	"""
	This method just simply convolves the images as if there is no padding. 
	The padding could be added prior to produce intended result image.
	"""
	# Get the size information
	size_x, size_y = target.shape[:2]
	kernel_x, kernel_y = kernel.shape[:2]
	
	# Create an empty convolved image template
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
			convolved_image[x - kernel_x // 2, y - kernel_y //2] = np.sum(np.multiply(target_window, kernel))
	
	return convolved_image
	
def naive_matching(target, template, verbose = True):
	# Get the size information
	size_y, size_x = target.shape[:2]
	template_y, template_x = template.shape[:2]
	
	# Create an empty convolved image template
	match_y = size_y - template_y + 1
	match_x = size_x - template_x + 1
	match_result = np.zeros((match_y, match_x))
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
	
	
	#naive_convolve(g_minsoo, sobel)
	#naive_convolve(g_minsoo, sobel2)
	coord, match_img = naive_matching(g_minsoo, g_ear, False)
	print(coord)
	plt.subplot(131)
	plt.imshow(g_minsoo,'gray')
	plt.subplot(132)
	plt.imshow(g_ear,'gray')
	plt.subplot(133)
	plt.imshow(match_img,'gray')
	plt.show()
	
	