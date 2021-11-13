import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import numpy as np
import sys
from utils import *

def res_block(x_in,num_filters):
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization()(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = PReLU(shared_axes=[1])(x)
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = PReLU(shared_axes=[1])(x)
    x = Add()([x_in, x])
    return x

def upsample(x_in, num_filters):
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization()(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = PReLU(shared_axes=[1])(x)
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Activation('linear')(x)
    return x

class cGAN_NETWORK():
    def __init__(self, n_qoi=None, n_data=None, n_z=None, rep_size=None, alpha=[1., 0.1, 1.],mu_sig=None,
                       g_n_resBlocks=None, g_n_filters=None,d_firstNFilt=None):
   
        self.n_qoi         = n_qoi
        self.n_data        = n_data
        self.n_z           = n_z
        #self.generator     = self.buildGenerator(n_data,n_qoi,n_z,n_resBlocks=8,n_filters=16)
        self.generator     = self.buildGenerator(n_data,n_qoi,n_z,n_resBlocks=g_n_resBlocks,n_filters=g_n_filters)
        #self.discriminator = self.buildDiscriminator(n_data,firstNFilt=8)
        self.discriminator = self.buildDiscriminator(n_data,firstNFilt=d_firstNFilt)
        self.generator.summary()
        self.discriminator.summary()
        self.alpha         = alpha   
        self.mu_sig        = mu_sig
 
    def buildGenerator(self, n_data, n_qoi, n_z, n_resBlocks=16, n_filters=64):
        if n_data%n_z > 0:
            print('n_data should be divisible by n_z')
            print('n_data =',n_data)
            print('n_z =',n_z)
            sys.exit()        
    
        noise_in = Input(shape=(n_z,1))
        qoi_in = Input(shape=(n_qoi,1))
        
        # Increase dimension of conditional variable path
        x_q  = Flatten()(qoi_in)
        x_q  = Dense(n_z)(x_q)
        #x_q  = LeakyReLU(alpha=0.2)(x_q) 
        x_q  = PReLU(shared_axes=[1])(x_q) 
        x_q  = Reshape((n_z,1))(x_q) 
     
        # Merge noise and conditional variable path
        x = Concatenate(axis=2)([x_q, noise_in])
      
        # B residual blocks
        x = Conv1D(n_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        #x = x_1 = LeakyReLU(alpha=0.2)(x) 
        x = x_1 = PReLU(shared_axes=[1])(x) 

        for _ in range(n_resBlocks):
            x = res_block(x, n_filters)
 
        x = Conv1D(n_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = PReLU(shared_axes=[1])(x)
        x = Add()([x_1, x])
    
        # Get the correct final size
        x = upsample(x, n_data//n_z)
        x = Reshape((n_data,1))(x)
            

        return Model([qoi_in,noise_in],x)
    
    def buildDiscriminator(self, n_data, firstNFilt=32):
         x_in = Input(shape=(n_data,1))
         x = Conv1D(firstNFilt,kernel_size=3,strides=1)(x_in)
         x = PReLU(shared_axes=[1])(x)
         x = Conv1D(firstNFilt,kernel_size=3,strides=2)(x)
         x = PReLU(shared_axes=[1])(x)
         for ifactor in range(3):
             factor = ifactor+2
             nFilt = (firstNFilt*factor)
             x = Conv1D(nFilt,kernel_size=3,strides=1)(x)
             x = PReLU(shared_axes=[1])(x)
             x = Conv1D(nFilt,kernel_size=3,strides=2)(x)
             x = PReLU(shared_axes=[1])(x)
         x = Flatten()(x)
         x = Dense(int(firstNFilt*32))(x)
         x = PReLU(shared_axes=[1])(x)
         #x = Dense(1,activation='sigmoid')(x)
         x = Dense(1)(x)
         
         return Model(x_in,x)

    def contentLoss(self,qoi,x_gen):
        x_gen_norescaled = x_gen*self.mu_sig[1][1] + self.mu_sig[1][0]
        mean_qoigen = tf.reduce_mean(x_gen_norescaled,axis=1,keepdims=True)
        qoigen = tf.reduce_mean(tf.square(x_gen_norescaled - mean_qoigen),axis=[1,2])
        qoi_norescaled = qoi * self.mu_sig[0][1] + self.mu_sig[0][0]
        return tf.square(qoigen-tf.reduce_mean(qoi_norescaled,axis=[1,2]))

    def generator_adversarialLoss(self,x_gen):
        disc_gen = self.discriminator(x_gen, training=False)
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_gen, labels=tf.ones_like(disc_gen))

    def discriminator_adversarialLoss(self,x_data,x_gen):
        disc_gen  = self.discriminator(x_gen,training=True)
        disc_data = self.discriminator(x_data,training=True)
        return tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=tf.concat([disc_data, disc_gen], axis=0),
                  labels=tf.concat([tf.ones_like(disc_data), tf.zeros_like(disc_gen)], axis=0)
               )

    def diversityLoss(self,qoi,x_gen,condMean_true,condVar_true,rep_size):
        qoi_untang   = tf.reshape(qoi,   [-1, rep_size, self.n_qoi,  1])
        x_gen_untang = tf.reshape(x_gen, [-1, rep_size, self.n_data, 1])

        qoi_untang_norescaled = qoi_untang * self.mu_sig[0][1] + self.mu_sig[0][0]
        x_gen_untang_norescaled = x_gen_untang * self.mu_sig[1][1] + self.mu_sig[1][0]

        condMean_gen, condVar_gen = tf.nn.moments(x_gen_untang_norescaled, axes=1)
        div_err_mean = tf.reduce_mean((condMean_true*self.mu_sig[1][1] - condMean_gen)**2, axis=[1, 2])
        div_err_var  = tf.reduce_mean(condVar_true*self.mu_sig[1][1] + condVar_gen
                               - 2*tf.multiply(tf.sqrt(condVar_true*self.mu_sig[1][1]), tf.sqrt(condVar_gen)), axis=[1, 2])
        divLoss = tf.repeat(div_err_mean+div_err_var, rep_size, axis=0)
        #return tf.reduce_mean(div_err_mean + div_err_var, axis=-1)        
        return divLoss

    def generator_Loss(self,qoi,x_gen,condMean_true,condVar_true,rep_size):
        g_con_loss = self.contentLoss(qoi,x_gen)
        g_adv_loss = self.generator_adversarialLoss(x_gen)
        g_div_loss = self.diversityLoss(qoi,x_gen,condMean_true,condVar_true,rep_size)
        #print('conloss =' , g_con_loss)
        #print('advloss =' , g_adv_loss)
        #print('divloss =' , g_div_loss)
        return self.alpha[0]*tf.reduce_mean(g_con_loss) + self.alpha[1]*tf.reduce_mean(g_adv_loss) + self.alpha[2]*tf.reduce_mean(g_div_loss)

    def discriminator_Loss(self,x_data,x_gen):
        d_adv_loss = self.discriminator_adversarialLoss(x_data,x_gen)
        return tf.reduce_mean(d_adv_loss)
