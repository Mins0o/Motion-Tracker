from Convolution import *
from FourierTransform import *
minsoo = (np.mean(plt.imread(r"D:\Dropbox\Workspace\03 Python\05 MotionTracker\Motion-Tracker\Img\Minsoo.jpg"), axis = 2)+0.5)/256
cMinsoo =(plt.imread(r"D:\Dropbox\Workspace\03 Python\05 MotionTracker\Motion-Tracker\Img\Minsoo.jpg")+0.5)/256
sobel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])