import cv2
import math
import numpy as np
import sys
import test

def getImage():
	if len(sys.argv) == 2:
		image = cv2.imread(sys.argv[1])
	elif len(sys.argv) < 2:
		cap = cv2.VideoCapture(0)
		for i in range(0,10):
			ret_val, image = cap.read()
		if ret_val:
			cv2.imwrite("capture.jpg", image)
	print("shape of original image",image.shape)
	return image

def showImage(image):
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def show(image):
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def reload(image):
	showImage(image)
	return image

def save(image):
	cv2.imwrite("new.jpg", image)

def smooth(image):
	def sliderHandler(n):
		if n != 0:
			global dst
			kernel = np.ones((n, n), dtype = np.float32) / (n * n)
			dst = cv2.filter2D(image, -1, kernel)
		cv2.imshow("Display window", dst)
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.createTrackbar("Smoothing", "Display window", 0, 255, sliderHandler)
	show(image)
	return dst

def cython(img):
    img = img.astype(np.int32)
    img1 = test.compute(img)
    img1 = np.array(img1, dtype=np.uint8)
    return img1

def cython_smoothing(img):
    def sliderHandler(n):
        img_copy = img.copy()
        for i in range(n):
            img_copy = cython(img_copy)
        cv2.imshow("Display window", img_copy)
    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Smoothing", "Display window", 0, 10, sliderHandler)
    cv2.imshow("Display window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def downSample(image):
	image = smooth(image)
	image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
	show(image)

def upSample(image):
	image = cv2.resize(image, (int(image.shape[1] * 2), int(image.shape[0] * 2)))
	image = smooth(image)
	show(image)

# computing the x-derivative
def xdrv(image):
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 5)
	# not sure
	norm = cv2.normalize(sobelx, np.zeros((800,800)), 0, 1, cv2.NORM_MINMAX)
	show(norm)

# computing the magnitude of image gradient
def mag(image):
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
	sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
	norm = cv2.normalize(sobelxy, np.zeros((800,800)), 0, 1, cv2.NORM_MINMAX)
	print(sobelxy)
	show(norm)

# plotting the image gradient vectors
def plot(image):
	def sliderHandler(n):
		new = image.copy()
		if n != 0:
			sobelx = cv2.Sobel(new, cv2.CV_64F, 1, 0, ksize = 5)
			sobely = cv2.Sobel(new, cv2.CV_64F, 0, 1, ksize = 5)
			for x in range(0, new.shape[0], n):
				for y in range(0, new.shape[1], n):
					grad_angle = math.atan2(sobely[x, y], sobelx[x, y])
					grad_x = int(x + n * math.cos(grad_angle))
					grad_y = int(y + n * math.sin(grad_angle))
					cv2.arrowedLine(new, (y, x), (grad_y, grad_x), (0, 0, 0))
		cv2.imshow("Display window", new)
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.createTrackbar("Value", "Display window", 0, 255, sliderHandler)
	show(image)

# detecting corners in the image using the OpenCV Harris corner detection function
def corner1(image):
	def sliderHandler(n):
		if n != 0:
			img = np.float32(image)
			dst = cv2.cornerHarris(img, n, 3, 0.04)
			dst = cv2.dilate(dst, None)
			img = cv2.merge((image, image, image))
			img[dst > 0.01 * dst.max()] = [0, 0, 255]
		cv2.imshow("Display window", img)
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.createTrackbar("Value", "Display window", 0, 255, sliderHandler)
	show(image)

# detecting corners using your own implementation of Harris corner detection algorithm
def corner2(image, k = 0.04, threshold = 100000):

	def sliderHandler(n):
		res = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
		if n != 0:
			offset = int(n / 2)
			y_range = image.shape[0] - offset
			x_range = image.shape[1] - offset

			dy, dx = np.gradient(image)
			Ixx = dx**2
			Ixy = dy*dx
			Iyy = dy**2

			for y in range(offset, y_range):
				for x in range(offset, x_range):
					# sliding window
					start_y = y - offset
					end_y = y + offset + 1
					start_x = x - offset
					end_x = x + offset + 1

					windowIxx = Ixx[start_y : end_y, start_x : end_x]
					windowIxy = Ixy[start_y : end_y, start_x : end_x]
					windowIyy = Iyy[start_y : end_y, start_x : end_x]

					# sum of squares of intensities of partial derevatives 
					Sxx = windowIxx.sum()
					Sxy = windowIxy.sum()
					Syy = windowIyy.sum()

					# determinant and trace of the matrix
					det = (Sxx * Syy) - (Sxy ** 2)
					trace = Sxx + Syy

					# r for Harris Corner equation
					r = det - k * (trace ** 2)

					if r > threshold:
						res[y,x] = (0, 0, 255)
		cv2.imshow("Display window", res)
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.createTrackbar("Value", "Display window", 0, 255, sliderHandler)
	show(image)

def program_desc():
	print ("'i': reload the original image")
	print ("'w': save the current image")
	print ("'s': smooth using OpenCV")
	print ("'S': smooth using your own implementation")
	print ("'D': downsample by a factor of 2")
	print ("'U': upsample by a factor of 2")
	print ("'x': compute the x-derivative")
	print ("'m': compute the magnitude of the image gradient")
	print ("'p': plot the image gradient vectors")
	print ("'c': detect corners in the image using the OpenCV Harris corner detection function")
	print ("'C': detect corners using your own implementation of Harris corner detection algorithm")
	print ("'h': display a short description of the program")

def main():
	image = getImage()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("shape of gray scale image", image.shape)
	img = image.copy()
	print("input key to Process image(press 'q' to quit): ")
	key = input()
	while key != 'q':
		print("input key to Process image(press 'q' to quit): ")
		if key == 'i':
			image = reload(img)
		elif key == 'w':
			save(image)
		elif key == 's':
			image = smooth(image)
		elif key == 'S':
			image = cython_smoothing(image)
			show(image)
		elif key == 'D':
			image = downSample(image)
		elif key == 'U':
			image = upSample(image)
		elif key == 'x':
			image = xdrv(image)
		elif key == 'm':
			image = mag(image)
		elif key == 'p':
			image = plot(image)
		elif key == 'c':
			image = corner1(image)
		elif key == 'C':
			image = corner2(image)
		elif key == 'h':
			program_desc()
		key = input()

if __name__ == '__main__':
	main()