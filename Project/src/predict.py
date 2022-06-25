from numpy import reshape
import tensorflow as tf
import cv2
import sys


if len(sys.argv) == 3:
    haze_img_path = sys.argv[1]
    model_path = sys.argv[2]
elif len(sys.argv) < 3:
    haze_img_path = 'download.jpg'
    model_path = "g_model.h5"




#haze_img_path = 'download.jpg'
haze_img = tf.keras.preprocessing.image.load_img(haze_img_path,target_size=(256,256))
haze_img = tf.keras.preprocessing.image.img_to_array(haze_img)
haze_img = haze_img/255
#print(haze_img.shape)
haze_img = haze_img.reshape(1,256,256,3)
#print(haze_img.shape)
#model_path = "g_model_tiramisu_epoch_10.h5"
model = tf.keras.models.load_model(model_path)
img = model.predict(haze_img)

img = img.reshape(256,256,3)
img = img*255
img=img[:,:,::-1]
cv2.imwrite("out.jpg",img)
print("Saved as out.jpg")