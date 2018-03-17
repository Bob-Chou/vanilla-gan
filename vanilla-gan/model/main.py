from vanilla_gan import *

def main():
	'''
	A quick startup for this model
	'''
	# use CNN to construct the GAN
	# startup = VanillaGAN(epoch_num=5, model='cnn', layers_num=2, d_hidden=[32, 64], g_hidden=[1024, 128], use_model=False)
	# use DNN to construct the GAN
	startup = VanillaGAN(epoch_num=10, model='dnn', layers_num=2, g_hidden=[1024, 1024], d_hidden=[256, 256], use_model=False)

	# train the model
	startup.train(print_every=100, plot_every=-1, log_every=200)

	# test the model and view the results
	show_images(startup.sample(batch_size = 16))

if __name__ == '__main__':
	main()

