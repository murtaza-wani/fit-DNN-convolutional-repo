#ifndef GLOBAL_CONSTANTS_H
#define GLOBAL_CONSTANTS_H

namespace globalconstants

{
	// uncomment if task is MNIST or Fashion-MNIST:
	const int M = 784;	   // number of features = number of input nodes
	const int M_root = 28; // length =  width of the 3D Image
	const int P = 10;	   // number of classes = number of output nodes
	const int number_of_training_batches = 6;
	const int first_conv_input_channels = 1; // number of channels in the input image
	const int first_conv_output_channels = 3;
	const int second_conv_input_channels = 3; // same as first conv output channels
	const int second_conv_output_channels = 3;
	const int K = 3; // size of kernel
	const int training_batch_size = 1000;
	const int test_batch_size = 1000;
	const int skip = 0;

	// uncomment if task is CIFAR-10:
	/* const int M = 3072; // number of features = number of input nodes
	const int M_root = 32; // size of length=width of 2-dimensional image
	const int P = 10;  // number of classes = number of output nodes
	const int number_of_training_batches = 5;
	const int first_conv_input_channels = 3;  // number of channels in the input image
	const int first_conv_output_channels = 3;
	const int second_conv_input_channels = 3; // same as first conv output channels
	const int second_conv_output_channels = 3;
	const int K = 3; // size of kernel
	const int training_batch_size = 1000;
	const int test_batch_size = 1000;
	const int skip = 0; */

	// uncomment if task is SVHN:
	// const int M = 3072;  // number of features = number of input nodes
	// const int P = 10;  // number of classes = number of output nodes
	// const int number_of_training_batches = 6;
	// const int training_batch_size = 12209;
	// const int test_batch_size = 26032;

	// uncomment if task is CIFAR-100-coarse:
	// const int M = 3072;  // number of features = number of input nodes
	// const int P = 20;  // number of classes = number of output nodes
	// const int number_of_training_batches = 5;
	// const int training_batch_size = 10000;
	// const int test_batch_size = 10000;
}

#endif
