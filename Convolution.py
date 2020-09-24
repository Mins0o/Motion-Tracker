import numpy as np
import matplotlib.pyplot as plt

def naive_convolve(target, kernel):

	size_x, size_y = target.shape[:2]
	kernel_x, kernel_y = kernel.shape[:2]
	# target shape validi_Aty check
	
	
	convolved_image = np.zeros((size_x - kernel_x + 1,size_y - kernel_y + 1, 3))
	# one channel
	#if target.shape[2] == 1:
	for x in range(kernel_x // 2, size_x - kernel_x // 2):
		for y in range(kernel_y // 2, size_y - kernel_y // 2):
			target_window = target[x - kernel_x // 2 : x + kernel_x // 2 + 1, y - kernel_y // 2 : y + kernel_y //2 + 1]
			convolved_image[x - kernel_x // 2, y - kernel_y // 2] = np.sum(np.multiply(target_window, kernel))
	
	plt.imshow(convolved_image,'gray')
	plt.show()
	
	if target.shape[2] == 3:
		# separate channels
		channel1 = target[:,:,0]
		channel2 = np.zeros(1)
		
def naive_single_channel():
	pass