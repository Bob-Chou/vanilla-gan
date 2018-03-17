from utils import *
from tensorflow.examples.tutorials.mnist import input_data

class VanillaGAN(object):
	'''docstring for VanillaGAN'''
	def __init__(self, model, layers_num, d_hidden, g_hidden, noise_dim=96, learning_rate=1e-3, beta1=0.5, minibatch_size=128, epoch_num=5, use_summary=True, use_save=True, use_model=False):
		super(VanillaGAN, self).__init__()
		self.data = input_data.read_data_sets('../datasets/MNIST_data', one_hot=False)
		self.image_dim = 784
		self.image_dim_sqrt = int(np.ceil(np.sqrt(self.image_dim)))
		self.model = model
		self.layers_num = layers_num
		if len(g_hidden) != layers_num or len(d_hidden) != layers_num:
			raise ValueError('input g_hidden and d_hidden should match the layers_num!')
		self.hidden_dim={'d': d_hidden, 'g': g_hidden}
		self.noise_dim = noise_dim
		self.minibatch_size = minibatch_size
		self.epoch_num = epoch_num
		self.solver = {
			'd': tf.train.AdamOptimizer(learning_rate, beta1),
			'g': tf.train.AdamOptimizer(learning_rate, beta1)
		}
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.use_summary = use_summary
		self.use_save = use_save
		self.use_model = use_model
		# if use tensorboard to summary
		if self.use_summary:
			self.log_path = '../log'
		# if use the saved model
		if self.use_model:
			try:
				self.saver = tf.train.import_meta_graph('../bin/vanilla_gan_model.meta')
				self.graph = tf.get_default_graph()
				self.session = tf.Session(graph = self.graph)
				self.saver.restore(self.session, tf.train.latest_checkpoint('../bin'))
			except OSError:
				raise OSError('cannot find the model, please use \'use_model==False\' to initialze a VanillaGAN')
			# Restore the ops and tensors
			self.inputs = self.graph.get_tensor_by_name('vanilla_gan/real:0')
			self.noise_batch = self.graph.get_tensor_by_name('vanilla_gan/noise_batch:0')
			self.products = self.graph.get_tensor_by_name('vanilla_gan/generator/products/Tanh:0')
			self.loss = {
				'd': self.graph.get_tensor_by_name('vanilla_gan/add:0'),
				'g': self.graph.get_tensor_by_name('vanilla_gan/mul_3:0')
			}
			self.params = {
				'd': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla_gan/discriminator'),
				'g': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla_gan/generator')
			}
			self.train_step = {
				'd': self.graph.get_operation_by_name('vanilla_gan/Adam'),
				'g': self.graph.get_operation_by_name('vanilla_gan/Adam_1')
			}
			if self.use_summary:
				tf.summary.scalar('sum_d_loss', self.loss['d'])
				tf.summary.scalar('sum_g_loss', self.loss['g'])
				tf.summary.image('sum_products', tf.reshape(self.products[:1], (1, self.image_dim_sqrt, self.image_dim_sqrt, -1)))
				self.summary = tf.summary.merge_all()
			if self.use_save:
				self.saver = tf.train.Saver()


		else:
			with self.graph.as_default():
				with tf.variable_scope('vanilla_gan', reuse=tf.AUTO_REUSE) as scope:
					# placeholder for images from the training dataset
					self.inputs = tf.placeholder(tf.float32, (None, self.image_dim), name='real')
					self.noise_batch = tf.placeholder(tf.int32, (None), name='noise_batch')
					# random noise fed into our generator
					z = tf.random_uniform((self.noise_batch, self.noise_dim), minval=-1, maxval=1, name='noise')
					# now define the gan
					self.products = generator(z, self.image_dim, self.model, self.layers_num, self.hidden_dim['g'])
					real = discriminator(image_normal(self.inputs), self.model, self.layers_num, self.hidden_dim['d'])
					fake = discriminator(self.products, self.model, self.layers_num, self.hidden_dim['d'])
					# compute the loss
					d_loss, g_loss = gan_loss(real, fake)
					# record some operations
					self.loss = {
						'd': d_loss,
						'g': g_loss
					}
					# Get the list of parameters of the discriminator and generator for training
					self.params = {
						'd': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla_gan/discriminator'),
						'g': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla_gan/generator')
					}
					# Define the training operation
					self.train_step = {
						'd': self.solver['d'].minimize(self.loss['d'], var_list = self.params['d']),
						'g': self.solver['g'].minimize(self.loss['g'], var_list = self.params['g'])		
					}
					# (if) summary the training process
					if self.use_summary:
						tf.summary.scalar('sum_d_loss', self.loss['d'])
						tf.summary.scalar('sum_g_loss', self.loss['g'])
						tf.summary.image('sum_products', tf.reshape(self.products[:1], (1, self.image_dim_sqrt, self.image_dim_sqrt, -1)))
						self.summary = tf.summary.merge_all()
					# (if) save the model
					if self.use_save:
						self.saver = tf.train.Saver()
				self.session.run(tf.global_variables_initializer())

	def train(self, epoch_num=None, print_every=100, plot_every=500, log_every=200):
		summary_writer = tf.summary.FileWriter(self.log_path, self.session.graph)
		# Get the total training iterations
		if epoch_num == None:
			epoch_num = self.epoch_num
		total_iter = self.data.train.num_examples * epoch_num // self.minibatch_size
		# Now start training
		for t in range(total_iter):
			minibatch, _ = self.data.train.next_batch(self.minibatch_size)
			d_loss, _ = self.session.run([self.loss['d'], self.train_step['d']], feed_dict={self.inputs: minibatch, self.noise_batch: self.minibatch_size})
			g_loss, _ = self.session.run([self.loss['g'], self.train_step['g']], feed_dict={self.noise_batch: self.minibatch_size})
			if print_every > 0 and t % print_every == 0:
				print('training: {}/{}, discriminator: {:.4}, generator: {:.4}'.format(t, total_iter, d_loss, g_loss))
			if plot_every > 0 and  t % plot_every == 0:
				fig = show_images(self.session.run(self.products, feed_dict={self.noise_batch: 16}))
			if log_every > 0 and t % log_every == 0 and self.use_summary:
				summary_writer.add_summary(self.session.run(self.summary, feed_dict={self.inputs: minibatch, self.noise_batch: self.minibatch_size}), t)
		if self.use_save:
			self.saver.save(self.session, '../bin/vanilla_gan_model')
	def sample(self, batch_size=1):
		'''
		Sample the generator

		Inputs:
			batch_size
		Output:
			sample_image with the same size as batch_size
		'''
		if batch_size > 50 or batch_size < 1:
			raise ValueError('expected batch size ranged from [1, 50], but obtained {}'.format(batch_size))
		else:
			return image_denorm(self.session.run(self.products, feed_dict={self.noise_batch: batch_size}))