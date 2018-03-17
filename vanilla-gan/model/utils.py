import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
'''
borrow from http://cs231n.github.io/assignments2017/assignment3/
'''
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    plt.show()
    return

def image_normal(x):
	'''
	Scale the image ranged from [0, 1] to [-1, 1]
	'''
	return 2 * x - 1

def image_denorm(x):
	'''
	Restore the image scaled in [-1, 1] to [0, 1]
	'''
	return (x + 1) / 2

def discriminator(x, model, layers, hidden_dim):
	'''
	Define the achitecture of the discriminator

	Inputs:
		- x: TensorFlow tensor of flattened input images, shape [batch_size, 784]
		- model: Model of the discriminator, 'vanilla' or 'convolutional'
		- layers: Layers of the discriminator
		- hidden_dim: number of the units in hidden layers

	Returns:
		- logits: 
			TensorFlow Tensor with shape [batch_size, 1], containing the score 
			for an image being real for each input image.
	'''
	if model == 'dnn':
		with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
			for l in range(layers):
				x = tf.layers.dense(inputs = x, units = hidden_dim[l], use_bias = True)
				x = tf.nn.leaky_relu(x, 0.01)
			# the output layer
			logits = tf.layers.dense(inputs = x, units = hidden_dim[l], use_bias = True, name='logits')
			return logits
	elif model == 'cnn':
		with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
			x = tf.reshape(x, (-1, 28, 28, 1))
			for l in range(layers):
				x = tf.layers.conv2d(x, filters=hidden_dim[l], kernel_size=5, strides=1)
				x = tf.nn.leaky_relu(x, 0.01)
				x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
			x = tf.layers.flatten(x)
			x = tf.layers.dense(x, 4*4*64)
			x = tf.nn.leaky_relu(x, 0.01)
			logits = tf.layers.dense(x, 1, name='logits')
			return logits

def generator(x, img_dim, model, layers, hidden_dim):
	'''
	Define the architecture of the generator

	Inputs:
		- x: Tensorflow tensor of flattened input noise, shape [batch_size, noise_dim]
		- img_dim: Shape of the target output, supposed to be the same shape as the ground truth image
		- model: Model of the generator, 'vanilla' or 'convolutional'
		- layers: Layers of the generator
		- hidden_dim: number of the units in hidden layers

	Outputs:
		- products: The generated image, with the same shape as the ground truth image [batch_size, 784]
	'''
	if model == 'dnn':
		with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
			for l in range(layers):
				x = tf.layers.dense(inputs = x, units = hidden_dim[l], use_bias=True, activation=tf.nn.relu)
			products = tf.layers.dense(inputs = x, units = img_dim, use_bias=True, activation=tf.tanh, name='products')
			return products
	elif model == 'cnn':
		img_dim = int(np.sqrt(img_dim))
		with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
			for l in range(layers-1):
				x = tf.layers.dense(inputs = x, units = hidden_dim[l], activation=tf.nn.relu)
				x = tf.layers.batch_normalization(x, training=True)
			# The last dense layer
			x = tf.layers.dense(inputs = x, units = img_dim//4 * img_dim//4 * hidden_dim[-1], activation=tf.nn.relu)
			x = tf.layers.batch_normalization(x, training=True)
			x = tf.reshape(x, (-1, img_dim//4, img_dim//4, hidden_dim[-1]))
			# The de-conv layer
			x = tf.layers.conv2d_transpose(x, 64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu)
			x = tf.layers.batch_normalization(x, training=True)        
			products = tf.layers.conv2d_transpose(x, 1, kernel_size=4, strides=2, padding='SAME', activation=tf.tanh, name='products')
			return products

def gan_loss(real, fake):
	'''
	Define the loss of vanilla-GAN. Here using ls-loss instead of the original cross-entropy loss。

	Inputs:
		- real: ground truth images
		- fake: synthesized images from generator

	Outputs:
		- d_loss: loss of the discriminator
		- g_loss: loss of the generator
	'''
	d_loss = 0.5 * tf.reduce_mean(tf.pow(real - 1, 2)) + 0.5 * tf.reduce_mean(tf.pow(fake, 2))
	g_loss = 0.5 * tf.reduce_mean(tf.pow(fake - 1, 2))
	return d_loss, g_loss

def sample_noise(batch_size, noise_dim):
	'''
	Sample the noise from a uniform distribution
	'''
	ans =  tf.random_uniform((batch_size, noise_dim), minval=-1, maxval=1, name='noise')
	return ans


