
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def discriminator(img_shape):
    haze_img = tf.keras.Input(shape=img_shape)
    img = tf.keras.Input(shape=img_shape)

    merge = layers.Concatenate()([haze_img,img])
    layer_conv_1 = layers.Conv2D(64,(3,3),strides=(2,2),padding="same")(merge)
    layer_relu_1 = layers.LeakyReLU(0.2)(layer_conv_1)

    layer_conv_2 = layers.Conv2D(128,(3,3),strides=(2,2),padding="same")(layer_relu_1)
    layer_batchnorm_1 = layers.BatchNormalization()(layer_conv_2)
    layer_relu_2 = layers.LeakyReLU(0.2)(layer_batchnorm_1)
    
    layer_conv_3 = layers.Conv2D(256,(3,3),strides=(2,2),padding="same")(layer_relu_2)
    layer_batchnorm_2 = layers.BatchNormalization()(layer_conv_3)
    layer_relu_3 = layers.LeakyReLU(0.2)(layer_batchnorm_2)

    layer_conv_4 = layers.Conv2D(512,(3,3),strides=(2,2),padding="same")(layer_relu_3)
    layer_batchnorm_3 = layers.BatchNormalization()(layer_conv_4)
    layer_relu_4 = layers.LeakyReLU(0.2)(layer_batchnorm_3)

    layer_conv_5 = layers.Conv2D(512,(3,3),padding="same")(layer_relu_4)
    layer_batchnorm_4 = layers.BatchNormalization()(layer_conv_5)
    layer_relu_5 = layers.LeakyReLU(0.2)(layer_batchnorm_4)

    layer_conv_6 = layers.Conv2D(1,(3,3),padding="same")(layer_relu_5)
    out = layers.Activation('sigmoid')(layer_conv_6)

    model = tf.keras.models.Model([haze_img,img],out)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

def dense_block(x,r=4,f=32):
    for i in range(r):
        y = bn_rel_con(x,f=4*f)
        y = bn_rel_con(x,f,k=3)
        x = layers.Concatenate()([y,x])
    return x

def bn_rel_con(x,f=32,k=1):
    bn = layers.BatchNormalization()(x, training=True)
    relu = layers.ReLU()(bn)
    con = layers.Conv2D(f,(k,k),strides=(1,1),padding="same" )(relu)
    return con

def transition_down(x):
    x = bn_rel_con(x)
    drop = layers.Dropout(0.2)(x, training=True)
    pool = layers.AveragePooling2D(padding='same')(drop)
    return pool

def transition_up(x):
    x = layers.Conv2DTranspose(32,(3,3),strides=(2,2),padding='same')(x)
    return x


def generator(img_shape):
    haze_img = tf.keras.Input(shape=img_shape)

    layer_conv_1 = layers.Conv2D(32,(3,3),strides=(2,2),padding="same")(haze_img)

    #transition down
    d1 = dense_block(layer_conv_1)
    
    td1 = transition_down(d1)
    d2 = dense_block(td1)

    td2 = transition_down(d2)
    d3 = dense_block(td2)

    td3 = transition_down(d3)
    d4 = dense_block(td3)

    td4 = transition_down(d4)
    d5 = dense_block(td4)

    td5 = transition_down(d5)

    #botelneck
    db = dense_block(td5,r=15)

    #transition up
    tu1 = transition_up(db)
    concat1 = layers.Concatenate()([tu1,d5])

    du1 = dense_block(concat1)
    tu2 = transition_up(du1)
    concat2 = layers.Concatenate()([tu2,d4])

    du2 = dense_block(concat2)
    tu3 = transition_up(du2)
    concat3 = layers.Concatenate()([tu3,d3])

    du3 = dense_block(concat3)
    tu4 = transition_up(du3)
    concat4 = layers.Concatenate()([tu4,d2])

    du4 = dense_block(concat4)
    tu5 = transition_up(du4)
    concat5 = layers.Concatenate()([tu5,d1])
    
    du5 = dense_block(concat5)

    out = layers.Conv2DTranspose(img_shape[2],(3,3), strides=(2,2), padding="same")(du5)
    out_img = layers.Activation('sigmoid')(out)

    model = tf.keras.models.Model(haze_img,out_img)

    return model

def generator_Unet(img_shape):
    haze_img = tf.keras.Input(shape=img_shape)

    layer_conv_1 = layers.Conv2D(64,(3,3),strides=(2,2),padding="same")(haze_img)
    layer_relu_1 = layers.LeakyReLU(0.2)(layer_conv_1)

    layer_conv_2 = layers.Conv2D(128,(3,3),strides=(2,2),padding="same")(layer_relu_1)
    layer_batchnorm_1 = layers.BatchNormalization()(layer_conv_2, training=True)
    layer_relu_2 = layers.LeakyReLU(0.2)(layer_batchnorm_1)

    layer_conv_3 = layers.Conv2D(256,(3,3),strides=(2,2),padding="same")(layer_relu_2)
    layer_batchnorm_2 = layers.BatchNormalization()(layer_conv_3, training=True)
    layer_relu_3 = layers.LeakyReLU(0.2)(layer_batchnorm_2)

    layer_conv_4 = layers.Conv2D(512,(3,3),strides=(2,2),padding="same")(layer_relu_3)
    layer_batchnorm_3 = layers.BatchNormalization()(layer_conv_4, training=True)
    layer_relu_4 = layers.LeakyReLU(0.2)(layer_batchnorm_3)

    layer_conv_5 = layers.Conv2D(512,(3,3),strides=(2,2),padding="same")(layer_relu_4)
    layer_batchnorm_4 = layers.BatchNormalization()(layer_conv_5, training=True)
    layer_relu_5 = layers.LeakyReLU(0.2)(layer_batchnorm_4)

    layer_conv_6 = layers.Conv2D(512,(3,3),strides=(2,2),padding="same")(layer_relu_5)
    layer_batchnorm_5 = layers.BatchNormalization()(layer_conv_6, training=True)
    layer_relu_6 = layers.LeakyReLU(0.2)(layer_batchnorm_5)

    layer_conv_7 = layers.Conv2D(512,(3,3),strides=(2,2),padding="same")(layer_relu_6)
    layer_batchnorm_6 = layers.BatchNormalization()(layer_conv_7, training=True)
    layer_relu_7 = layers.LeakyReLU(0.2)(layer_batchnorm_6)


    layer_conv_8 = layers.Conv2D(1024,(3,3),strides=(2,2),padding="same")(layer_relu_7)
    layer_activation = layers.Activation('relu')(layer_conv_8)


    layer_conv_9 = layers.Conv2DTranspose(512,(3,3),strides=(2,2),padding="same")(layer_activation)
    layer_batchnorm_7 = layers.BatchNormalization()(layer_conv_9, training=True)
    layer_dropout_1 = layers.Dropout(0.5)(layer_batchnorm_7, training=True)
    layer_concat_1 = layers.Concatenate()([layer_relu_7,layer_dropout_1])
    layer_activation_2 = layers.Activation("relu")(layer_concat_1)

    layer_conv_10 = layers.Conv2DTranspose(512,(3,3),strides=(2,2),padding="same")(layer_activation_2)
    layer_batchnorm_8 = layers.BatchNormalization()(layer_conv_10, training=True)
    layer_dropout_2 = layers.Dropout(0.5)(layer_batchnorm_8, training=True)
    layer_concat_2 = layers.Concatenate()([layer_relu_6,layer_dropout_2])
    layer_activation_3 = layers.Activation("relu")(layer_concat_2)

    layer_conv_11 = layers.Conv2DTranspose(512,(3,3),strides=(2,2),padding="same")(layer_activation_3)
    layer_batchnorm_9 = layers.BatchNormalization()(layer_conv_11, training=True)
    layer_dropout_3 = layers.Dropout(0.5)(layer_batchnorm_9, training=True)
    layer_concat_3 = layers.Concatenate()([layer_relu_5,layer_dropout_3])
    layer_activation_4 = layers.Activation("relu")(layer_concat_3)

    layer_conv_12 = layers.Conv2DTranspose(512,(3,3),strides=(2,2),padding="same")(layer_activation_4)
    layer_batchnorm_10 = layers.BatchNormalization()(layer_conv_12, training=True)
    layer_concat_4 = layers.Concatenate()([layer_relu_4,layer_batchnorm_10])
    layer_activation_5 = layers.Activation("relu")(layer_concat_4)
    
    layer_conv_13 = layers.Conv2DTranspose(256,(3,3),strides=(2,2),padding="same")(layer_activation_5)
    layer_batchnorm_11 = layers.BatchNormalization()(layer_conv_13, training=True)
    layer_concat_5 = layers.Concatenate()([layer_relu_3,layer_batchnorm_11])
    layer_activation_6 = layers.Activation("relu")(layer_concat_5)

    layer_conv_14 = layers.Conv2DTranspose(128,(3,3),strides=(2,2),padding="same")(layer_activation_6)
    layer_batchnorm_12 = layers.BatchNormalization()(layer_conv_14, training=True)
    layer_concat_6 = layers.Concatenate()([layer_relu_2,layer_batchnorm_12])
    layer_activation_7 = layers.Activation("relu")(layer_concat_6)

    layer_conv_15 = layers.Conv2DTranspose(64,(3,3),strides=(2,2),padding="same")(layer_activation_7)
    layer_batchnorm_13 = layers.BatchNormalization()(layer_conv_15, training=True)
    layer_concat_7 = layers.Concatenate()([layer_relu_1,layer_batchnorm_13])
    layer_activation_8 = layers.Activation("relu")(layer_concat_7)

    out = layers.Conv2DTranspose(img_shape[2],(3,3), strides=(2,2), padding="same")(layer_activation_8)
    out_img = layers.Activation('sigmoid')(out)

    model = tf.keras.models.Model(haze_img,out_img)

    return model

def define_gan(g_model, d_model, img_shape):
	for layer in d_model.layers:
		if not isinstance(layer, layers.BatchNormalization):
			layer.trainable = False
	haze_img = tf.keras.Input(shape=img_shape)
	img = g_model(haze_img)
	d_out = d_model([haze_img, img])
	model = tf.keras.models.Model(haze_img, [d_out, img])
	opt = tf.keras.optimizers.Adam(lr=0.0002)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model


def real_samples(dataset,patch_shape):
    haze_img,img = dataset
    n = np.random.randint(0,haze_img.shape[0],1)
    haze_img_sample = haze_img[n]
    img_sample = img[n]
    y = np.ones((1,patch_shape,patch_shape,1))
    return [haze_img_sample,img_sample], y

def fake_sample(g_model,sample,patch_shape):
    img_sample = g_model.predict(sample)
    y = np.zeros((len(img_sample),patch_shape,patch_shape,1))
    return img_sample, y

def train(d_model,g_model,gan_model,dataset,epoch):
    patch = d_model.output_shape[1]

    haze_img, img = dataset

    bat_per_epoch = len(haze_img)

    steps = bat_per_epoch*epoch

    for i in range(steps):
        [haze_img_real,img_real], y_real = real_samples(dataset,patch)
        haze_img_fake,img_fake = fake_sample(g_model,haze_img_real,patch)
        d1_loss_real = d_model.train_on_batch([haze_img_real,img_real], y_real)
        d1_loss_fake = d_model.train_on_batch([haze_img_real,haze_img_fake], img_fake)
        g_loss,_,_ = gan_model.train_on_batch(haze_img_real,[y_real,img_real])

        print('>%d, d1_loss_real[%.3f] d1_loss_fake[%.3f] g_loss[%.3f]' % (i+1, d1_loss_real, d1_loss_fake, g_loss))
    g_model.save('g_model.h5')