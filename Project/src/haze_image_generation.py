import cv2
import h5py
import numpy as np
import os
from skimage.transform import resize

with h5py.File('data.mat', 'r') as data:
        images = np.array(data['images'])
        depths = np.array(data['depths'])

images = images.transpose(0, 1, 3, 2)
depths = depths.transpose(2, 1, 0)
depths = (depths - np.min(depths, axis = (0, 1))) / np.max(depths, axis = (0, 1))
depths = ((1 - depths) * np.random.uniform(0.2, 0.4, size = (1449, ))).transpose(2, 0, 1)

for i in range(len(images)):
    #storing image in folder named image
    img = resize(images[i].transpose(1, 2, 0), (256, 256, 3), mode = 'reflect')
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join('image', str(i).zfill(5) + '.png'), img)
    
    #storing haze image in folder named hazed_image
    haze = (images[i] * depths[i]) + (1 - depths[i]) * np.ones_like(depths[i]) * 255
    haze = resize(haze.transpose(1, 2, 0), (256, 256, 3), mode = 'reflect')                 #alternative way by using cv2.resize
    haze = cv2.cvtColor(haze.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join('hazed_image', str(i).zfill(5) + '.png'), haze)