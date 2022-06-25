import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gans import discriminator,generator,define_gan,train, generator_Unet

arc = int(input("Enter 0 for Tiramisu as generator\nEnter 1 for UNet as generator\n"))
epoch = int(input("Number of epoch to train: "))
haze_img_list = list()
img_list = list()
haze_img_path = 'hazed_image/'
for i in os.listdir(haze_img_path):
    haze_img = tf.keras.preprocessing.image.load_img(haze_img_path+i,target_size=(256,256))
    haze_img = tf.keras.preprocessing.image.img_to_array(haze_img)
    haze_img_list.append(haze_img)
haze_img_list = np.asarray(haze_img_list)

img_path = 'image/'
for i in os.listdir(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path+i,target_size=(256,256))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img_list.append(img)
img_list = np.asarray(img_list)

img_shape = img_list.shape[1:]
if (arc == 0):
    print("\n\n************************************************************")
    print("\tTiramisu as generator in use")
    print("************************************************************\n\n")
    d_model = discriminator(img_shape)
    g_model = generator(img_shape)
    gan_model = define_gan(g_model,d_model,img_shape)

    haze_img_list = haze_img_list/255
    img_list = img_list/255

    dataset = [haze_img_list,img_list]

    train(d_model,g_model,gan_model,dataset,epoch)
else:
    print("\n\n************************************************************")
    print("\tUNet as generator in use")
    print("************************************************************\n\n")
    d_model = discriminator(img_shape)
    g_model = generator_Unet(img_shape)
    gan_model = define_gan(g_model,d_model,img_shape)

    haze_img_list = haze_img_list/255
    img_list = img_list/255

    dataset = [haze_img_list,img_list]

    train(d_model,g_model,gan_model,dataset,epoch)