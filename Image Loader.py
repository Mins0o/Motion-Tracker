from PIL import Image

def numpy_image(path):
	img = Image.open(path)
	return np.array(img)