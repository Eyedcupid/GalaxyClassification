import numpy as np

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.optimizers import Adam



def conv_layer(layer,filters,
               kernel,activation='relu',
               padding='same',strides=1):
  return tf.keras.layers.Conv2D(filters=filters,
                         kernel_size=kernel,
                         strides=strides,
                         activation=activation,
                         padding=padding)(layer)

def conv_transpose_layer(layer,filters,
               kernel,activation='relu',
               padding='same',strides=1):
  return tf.keras.layers.Conv2DTranspose(filters=filters,
                         kernel_size=kernel,
                         strides=strides,
                         activation=activation,
                         padding=padding)(layer)

def model(input_shape,latent_dim):
  #Encoder 
  model_input=tf.keras.Input(input_shape)
  layer=conv_layer(model_input,16,5,strides=2)
  layer=conv_layer(layer,32,3,strides=2)
  layer=conv_layer(layer,64,3,strides=2)
  layer=conv_layer(layer,128,3,strides=2)
  shape_before_flatten=layer.shape
  layer=tf.keras.layers.Flatten()(layer)
  mean=tf.keras.layers.Dense(latent_dim)(layer)
  var=tf.keras.layers.Dense(latent_dim)(layer)
  encoder_model=tf.keras.models.Model(model_input,[mean,var])
  
  #Decoder
  decoder_input=tf.keras.Input((latent_dim,))
  layer=tf.keras.layers.Dense(np.prod(shape_before_flatten[1:]),\
                              activation='relu')(decoder_input)
  layer=tf.keras.layers.Reshape(target_shape=shape_before_flatten[1:])(layer)
  layer=conv_transpose_layer(layer,128,3,strides=2)
  layer=conv_transpose_layer(layer,64,3,strides=2)
  layer=conv_transpose_layer(layer,32,3,strides=2)
  layer=conv_transpose_layer(layer,16,5,strides=2)
  layer=conv_transpose_layer(layer,3,3,activation='sigmoid')
  decoder_model=tf.keras.models.Model(decoder_input,layer)    

  #Reparameterization Trick
  mean,var=encoder_model(model_input)
  epsilon=tf.random.normal(shape=(tf.shape(var)[0],
                                  tf.shape(var)[1]))
  z=mean+tf.exp(var)*epsilon
  model_out=decoder_model(z)
  model=tf.keras.models.Model(model_input,model_out)
 
  #Reconstruction loss
  reconstruction_loss = K.sum(K.binary_crossentropy(model_input,
                                                    model_out), axis=[1, 2, 3])
  
  #KL div loss
  kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
  
  elbo = K.mean(reconstruction_loss + kl_loss)
  model.add_loss(elbo)

  # Add metrics for tracking
  model.add_metric(K.mean(reconstruction_loss), name="reconstruction_loss", aggregation="mean")
  model.add_metric(K.mean(kl_loss), name="kl_loss", aggregation="mean")
  model.add_metric(elbo, name="elbo_loss", aggregation="mean")

  return model,decoder_model,encoder_model
     
