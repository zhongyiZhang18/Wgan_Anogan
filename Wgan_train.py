from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras import backend as K
from keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import math
import pandas as pd





import keras.backend as K

import matplotlib.pyplot as plt

import sys
import pandas as pd
import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = slice_row
        self.img_cols = slice_col
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 10

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.0002)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.generator.load_weights('saved_model/whited_Wgenerator.h5')
        self.critic = self.build_critic()
        self.critic.load_weights('./saved_model/whited_Wdiscriminator.h5')



        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(10,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
 #keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
        inputs = Input(shape=(self.latent_dim,))
        fc1 = Dense(input_dim=10, units=128*21*15)(inputs)
        fc1 = BatchNormalization()(fc1)
        fc1 = Activation('relu')(fc1)
        fc2 = Reshape((21, 15, 128), input_shape=(128*21*15,))(fc1)
        up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
        conv1 = Conv2D(64, (3, 3), padding='same')(up1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(1, (5, 5), padding='same')(up2)
        outputs = Activation('tanh')(conv2)
        noise=inputs
        img=outputs
        return Model(noise, img)

    def build_critic(self):

        inputs = Input((84,60,1))
        conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)      
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        fc1 = Flatten()(pool2)
        outputs = Dense(1)(fc1)

        img = inputs
        validity = outputs

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):


    
        wg_loss_list = []
        wd_loss_list = []
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)
            
            
              
#             wg_loss_list.append(g_loss)
#             wd_loss_list.append(d_loss[0])
                
            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

#         wg_dict={"wg_loss" : wg_loss_list,"wd_loss" : wd_loss_list}#将列表a，b转换成字典
#         wg_df=pd.DataFrame(wg_dict)#将字典转换成为数据框    
#         wg_df.to_csv("whited_wg&wd_loss.csv")
            
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c,figsize=(21,15))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("generated_images/Wgan_generated_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=200, batch_size=32, sample_interval=100)
    wgan.generator.save_weights('saved_model/whited_Wgenerator.h5', True)
    wgan.critic.save_weights('saved_model/whited_Wdiscriminator.h5', True)