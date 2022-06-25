import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as plt2

def getImage():
	if len(sys.argv) == 2:
		image = cv2.imread(sys.argv[1])
	elif len(sys.argv) < 2:
		cap = cv2.VideoCapture(0)
		for i in range(0,10):
			ret_val, image = cap.read()
		if ret_val:
			cv2.imwrite("capture.jpg", image)
	return image


def hough_transform(image_1 , image, num_rho = 135, num_theta = 135 , count = 512):

  edge_image = cv2.Canny(image,threshold1=30, threshold2=100)
    
  figure = plt.figure(figsize=(12, 12))
  
  # Subplots
  subplot1 = figure.add_subplot(1, 5, 1)
  subplot1.imshow(image)
  subplot1.set_title('Original Image')
  
  subplot2 = figure.add_subplot(1, 5, 2)
  subplot2.imshow(edge_image, cmap="gray")
  subplot2.set_title('Edge Detection')
  
  subplot3 = figure.add_subplot(1, 5, 4)
  subplot3.set_facecolor((0, 0, 0))
  subplot3.set_title('Hough Transform')
  
  subplot4 = figure.add_subplot(1, 5, 5)
  subplot4.imshow(image)
  subplot4.set_title('Detected Lines on Image')
  
  subplot5 = figure.add_subplot(1, 5, 3)
  subplot5.imshow(image_1)
  subplot5.set_title('Gray scale image')
  
  height = edge_image.shape[0]
  width = edge_image.shape[1]
  hh = np.divide(height,2)
  wh = np.divide(width,2)
  
  # diagonal calculations for maximum values to be a finite number of possible values.
  
  diagonal_len = np.sqrt(np.square(height) + np.square(width))
  diagonal_theta = 135 / num_theta
  diagonal_rho = (2 * diagonal_len) / num_rho
  
  # theta and ρ should have specific range
  thetas = np.arange(45,135 , step = diagonal_theta)
  rhos = np.arange(-diagonal_len, diagonal_len , step = diagonal_rho)
  
  
  cos_t = np.cos(np.deg2rad(thetas))
  sin_t = np.sin(np.deg2rad(thetas))
  
  
  # accumulator for 2d array of theta and rho
  accumulator = np.zeros((len(rhos), len(thetas)))
  
   # hough transform and making of matrix theta n rho for line 
  for y in range(height):
     for x in range(width):
       if edge_image[y][x] != 0:
         edge_point = [y - hh, x - wh]
         rho_matrix,theta_matrix = [], [] 
         for theta_idx in range(len(thetas)):  
           rho_d = (edge_point[0] * cos_t[theta_idx]) + (edge_point[1] * sin_t[theta_idx])
           theta = thetas[theta_idx]
           rho_idx = np.argmin(np.abs(rhos - rho_d))  
           accumulator[rho_idx][theta_idx] += 1
           rho_matrix.append(rho_d)
           theta_matrix.append(theta) 
         subplot3.plot(theta_matrix ,rho_matrix, color="green", alpha=0.05)

  for x in range(accumulator.shape[0]):
     for y in range(accumulator.shape[1]):
         
       if accumulator[x][y] > count:
         rho = rhos[x]
         theta =thetas[y]
         
         a = np.cos(np.deg2rad(theta))
         b = np.sin(np.deg2rad(theta))
         
         c = (a * rho) + wh
         d = (b * rho) + hh
         x1 = int(c + 1000 * (-b))
         y1 = int(d + 1000 * (a))
         x2 = int(c - 1000 * (-b))
         y2 = int(d - 1000 * (a))
         subplot3.plot([theta], [rho], marker='o', color="red")
         subplot4.add_line(plt2.Line2D([x1, x2], [y1, y2] , color = "Red"))
  plt.savefig("generated_plot.png")       

    
def main():
    global image
    global filename
    image = getImage()
    image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Processing Line detection transformations")
    hough_transform(image_1, image)

if __name__ == '__main__':
    main()