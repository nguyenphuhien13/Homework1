# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    
	
	
	x = list(range(-3*math.floor(sigma),3*math.floor(sigma)+1))
	Gx = []

	for i in x:
		Gx.append((1/(math.sqrt(2*math.pi)*sigma))*math.exp(-i**2/(2*sigma**2)))
	
	Gx = Gx/np.sum(Gx)
	return Gx, x
	'''

	# multiply sigma by 6 and round up as the length of the array
	length = int(math.ceil(float(sigma) * 6))

	# if it is an even number, add 1 to make it odd
	if (length%2 == 0): length = length + 1

	# calcaute the mid point position
	center = length/2

	#x = [floor(-3.0*sigma+0.5):floor(3.0*sigma+0.5)];
	#x = [i ]
	# create an array of distance from -mid position to +mid position
	x = np.arange(-center, center + 1)

	# apply the Gaussian function to the array value
	Gx =  (1/(math.sqrt(2*math.pi)*sigma))*np.exp(-x**2/(2*sigma**2))

	# normalize the array so they sum up to 1
	Gx = Gx/np.sum(Gx)

    
	return Gx, x
	'''


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
	[Gx,x] = gauss(sigma)
	gauss2d_temp = Gx[np.newaxis]

	# calculate a transpose of the array
	gauss2d_temp_trans = gauss2d_temp.T

	# Gaussian 2D = convolution of 1D Gaussian with its transpose
	gauss2d_filter = conv2(gauss2d_temp, gauss2d_temp_trans)

	# apply convolution 
	smooth_img = conv2(img, gauss2d_filter, 'same')

	return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

	x = list(range(-3*math.floor(sigma),3*math.floor(sigma)+1))
	Dx = []

	for i in x:
		Dx.append((1/(math.sqrt(2*math.pi)*sigma**3))*i*np.exp(-i**2/(2*sigma**2)))
	
	Dx = Dx/np.sum(Dx)
	return Dx, x
	


def gaussderiv(img, sigma):
	
	G = gauss(sigma)
	D = gaussdx(sigma)
	img1 = conv2(img,D,'same');
	imgDx= conv2(img1,G,'same');
	img2 = conv2(img,G,'same');
	imgDy= conv2(img2,D,'same');

	return imgDx, imgDy

