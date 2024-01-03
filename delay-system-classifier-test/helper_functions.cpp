#include "helper_functions.h"

using namespace std;
using namespace arma;
using namespace globalconstants;

auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
mt19937 rng(seed);

#include <iostream>
#include <vector>
#include <armadillo>

double uniform(double a, double b)
{
	/*
	function returns random value from uniform distribution on [a, b]
	*/
	uniform_real_distribution<double> dis(a, b);
	return dis(rng);
}

// Matrix to give all the locations in the big sparse weight Matrix connecting two hidden convolutional layers where the convolutional weight (t,s,j,l) goes.
// get_kernel_weight_locations returns an matrix T (say) having two columns, where (T(i,0),T(i,1)) are locations in the big sparse weight matrix where convolutional weight (t,s,j,l) goes.
// We are assuming same padding so that image length and width remains unchanged
// t represents the kernel number and (s,j,l) represent the coordinates of weights in a 3-dimensional kernel t. In this way (t,s,j,l) are the weights involved in
// tranformation of one layer to next by convolution
// K=length of kernel = width of kernel
// M_root = width of image = height of image. We are using the assumption that width of image is same as it's height

Mat<int> get_kernel_weight_locations(int t, int s, int j, int l, int K, int M_root)
{

	// Calculate the number of valid locations where (j,l) is found
	int valid_locs = 0;
	int kernel_center = K / 2;
	int row_offset = j - kernel_center;
	int col_offset = l - kernel_center;

	for (int row_idx = 0; row_idx < M_root; row_idx++)
	{
		for (int col_idx = 0; col_idx < M_root; col_idx++)
		{
			int new_row = row_idx + row_offset;
			int new_col = col_idx + col_offset;
			if (new_row >= 0 && new_row < M_root && new_col >= 0 && new_col < M_root)
			{
				valid_locs++;
			}
		}
	}

	// Create matrix with size based on valid_locs
	Mat<int> kernel_tuples(valid_locs, 2, fill::zeros);

	int skipped = 0;

	for (int i = 0; i < M_root * M_root; i++)
	{
		int row_idx = i / M_root;
		int col_idx = i % M_root;

		int new_row = row_idx + row_offset;
		int new_col = col_idx + col_offset;

		if (new_row >= 0 && new_row < M_root && new_col >= 0 && new_col < M_root)
		{
			kernel_tuples(i - skipped, 0) = t * M_root * M_root + i;
			kernel_tuples(i - skipped, 1) = s * M_root * M_root + new_row * M_root + new_col;
		}
		else
		{
			skipped++;
		}
	}

	return kernel_tuples;
}

// Function to return the big Sparse weight Matrix given the kernels used for tranformation in 4D arma fields, output channels, input channels and M_root= image width = img height

arma::mat unroll_kernel_to_weight_mat(int output_channels, int input_channels, const arma::field<arma::cube> &kernels, int M_root)

{
	arma::mat fully_unrolled_K(output_channels * M_root * M_root, input_channels * M_root * M_root, arma::fill::zeros);

	for (int t = 0; t < output_channels; ++t)
	{
		for (int s = 0; s < input_channels; ++s)
		{
			for (int j = 0; j < K; ++j)
			{
				for (int l = 0; l < K; ++l)
				{
					arma::Mat<int> core = get_kernel_weight_locations(t, s, j, l, K, M_root);
					int numRows = core.n_rows;
					double value = kernels(t)(s, j, l);
					for (int i = 0; i < numRows; ++i)
					{
						// Assuming kernels(t)(s,j,l) gives the value at (t, s, j, l) in the kernels field
						fully_unrolled_K(core(i, 0), core(i, 1)) = value;
					}
				}
			}
		}
	}

	return fully_unrolled_K;
}

// Function to get all the non-zero diagonals in the big sparse weight matrix Computation of position of non-zero diagonals from the unrolled_matrix
vector<int> diagonal_indices(mat &unrolled)
{

	vector<int> non_zero_diag_indices;
	int numRows = unrolled.n_rows;
	int numCols = unrolled.n_cols;
	// Loop over the diagonals of the unrolled matrix
	for (int i = -numRows + 1; i < numCols - 1; i++)
	{
		// Extract the i-th diagonal of the unrolled matrix
		vec diag_vec = diagvec(unrolled, i);

		// Check if any element of the diagonal is non-zero
		if (any(diag_vec != 0))
		{
			// Add the index of the non-zero diagonal to the vector
			non_zero_diag_indices.push_back(-i);
		}
	}
	return non_zero_diag_indices;
}

double learning_rate(int epoch, int k, double eta_0, double eta_1)
{
	/*
	Function to compute the learning rate eta in dependence of the step k (and epoch).
	*/
	double k_total = epoch * (number_of_training_batches - 1) * training_batch_size + k;
	return min(eta_0, eta_1 / k_total);
}

void read_files(cube &train_images, mat &test_images, int (&train_labels)[number_of_training_batches][training_batch_size], int (&test_labels)[test_batch_size], string data_dir)
{
	/*
	Function to open the txt files in the data folder.
	Reads the images stored in these files to an armadillo cube resp. matrix
	and the labels to int arrays.

	Args:
	train_images: reference to arma::cube with 6 rows, 10000 cols, 784 slices (or corresponding numbers if task != MNIST).
				  To be filled with training image data.
				  Each row represents one training data batch,
				  each col of represents one image,
				  each slice represents one pixel.
	test_images:  reference to arma::matrix with 10000 rows, 784 cols (or corresponding numbers if task != MNIST).
				  To be filled with test image data.
				  Each row of represents one image,
				  each col (of a row) represents one pixel.
	train_labels: reference to an int array with size number_of_training_batches x training_batch_size (e.g. 6 x 10 for MNIST).
				  To be filled with labels (0, 1, ..., 9) of the training images.
	test_labels:  reference to an int array with size test_batch_size (e.g. 10000 for MNIST).
				  To be filled with labels of the test images.
	data_dir:     string
				  "../" + data_dir is the path to the directory containing the MNIST resp. Fashion-MNIST data set.
	*/

	// open training images and store data in armadillo cube normalized to [0, 1]
	// row is batch, col is image, slice is pixel
	fstream train_images_file;
	string train_image_string;
	for (int batch_index = 0; batch_index < number_of_training_batches; ++batch_index)
	{
		train_images_file.open("../" + data_dir + "/train_images_" + to_string(batch_index + 1) + ".txt", ios::in);
		for (int image_index = 0; image_index < training_batch_size; ++image_index)
		{
			getline(train_images_file, train_image_string);
			for (int pixel_index = 0; pixel_index < M; ++pixel_index)
			{
				string hex_code = train_image_string.substr(2 * pixel_index, 2);
				int int_pixel;
				stringstream s;
				s << hex << hex_code;
				s >> int_pixel;
				double double_pixel = int_pixel;
				train_images(batch_index, image_index, pixel_index) = double_pixel / 255;
			}
		}
		train_images_file.close();
	}

	// open test images file and store data in armadillo matrix normalized to [0, 1]
	// row is image, col is pixel
	fstream test_images_file;
	string test_image_string;
	test_images_file.open("../" + data_dir + "/test_images.txt", ios::in);
	for (int row = 0; row < test_batch_size; ++row)
	{
		getline(test_images_file, test_image_string);
		for (int col = 0; col < M; ++col)
		{
			string hex_code = test_image_string.substr(2 * col, 2);
			int int_pixel;
			stringstream s;
			s << hex << hex_code;
			s >> int_pixel;
			double double_pixel = int_pixel;
			test_images(row, col) = double_pixel / 255;
		}
	}
	test_images_file.close();

	// open training label files and store labels in int array
	for (int batch_index = 0; batch_index < number_of_training_batches; ++batch_index)
	{
		fstream train_labels_file;
		train_labels_file.open("../" + data_dir + "/train_labels_" + to_string(batch_index + 1) + ".txt", ios::in);
		string train_labels_string;
		getline(train_labels_file, train_labels_string);
		train_labels_file.close();
		if (data_dir == "data-CIFAR-100-coarse")
		{
			for (int image_index = 0; image_index < training_batch_size; ++image_index)
			{
				char c1 = train_labels_string[2 * image_index];
				char c2 = train_labels_string[2 * image_index + 1];
				train_labels[batch_index][image_index] = 10 * (c1 - '0') + (c2 - '0');
			}
		}
		else
		{
			for (int image_index = 0; image_index < training_batch_size; ++image_index)
			{
				char c = train_labels_string[image_index];
				train_labels[batch_index][image_index] = c - '0';
			}
		}
	}

	// open test label file and store labels in int array
	fstream test_labels_file;
	test_labels_file.open("../" + data_dir + "/test_labels.txt", ios::in);
	string test_labels_string;
	getline(test_labels_file, test_labels_string);
	test_labels_file.close();
	if (data_dir == "data-CIFAR-100-coarse")
	{
		for (int image_index = 0; image_index < test_batch_size; ++image_index)
		{
			char c1 = test_labels_string[2 * image_index];
			char c2 = test_labels_string[2 * image_index + 1];
			test_labels[image_index] = 10 * (c1 - '0') + (c2 - '0');
		}
	}
	else
	{
		for (int image_index = 0; image_index < test_batch_size; ++image_index)
		{
			char c = test_labels_string[image_index];
			test_labels[image_index] = c - '0';
		}
	}
}

void initialize_weights(vec &input_weights, field<cube> &first_conv_hidden_weights, field<cube> &second_conv_hidden_weights, mat &first_hidden_weights_mat, mat &second_hidden_weights_mat, mat &output_weights,

						double initial_input_weigt_radius, double initial_first_hidden_weigt_radius, double initial_second_hidden_weigt_radius, double initial_output_weigt_radius, bool save_to_file)
{
	/*
	Function to initialize weigths.

	Args:
	input_weights:      reference to arma::vec of size M
						Vec W^in. To be filled with the initial weights connecting the input layer to the first hidden layer
						to the first hidden layer (including the input bias weight).

	first_conv_hidden_weights:
						reference to arma field of #first_conv_output_channels x #first_conv_input_channels x K x K
						To be filled with intitial weights for the first convolution operation

	second_conv_hidden_weights:
						reference to arma field of #second_conv_output_channels x #second_conv_input_channels x K x K
						To be filled with intitial weights for the second convolution operation

	first_hidden_weights_mat:
						reference to arma mat of size (#first_conv_output_channels * M_root * M_root) x (#first_conv_input_channels * M_root * M_root)
						To be filled by unrolling the initialized first_conv_hidden_weights

	second_hidden_weights_mat:
						reference to arma mat of size (#second_conv_output_channels * M_root * M_root) x (#second_conv_input_channels * M_root * M_root)
						To be filled by unrolling the initialized second_conv_hidden_weights

	output_weights:     reference to arma::mat with size P x (#second_conv_output_channels * M_root * M_root + 1) (where P = 10).
						To be filled with the initial weights connecting the last hidden layer to the output layer..

	initial_input_weigt_radius:  double.
	initial_hidden_weigt_radius: double.
	initial_output_weigt_radius: double.

	*/

	// initial input weights
    input_weights = initial_input_weigt_radius * arma::ones<arma::vec>(M);

	// initial output weights
	output_weights = -initial_output_weigt_radius * mat(P, second_conv_output_channels * M_root * M_root + 1, fill::ones) + 2.0 * initial_output_weigt_radius * mat(P, second_conv_output_channels * M_root * M_root + 1, fill::randu);

	// initialize convolutional hidden weights
	// first convolutional hidden field
	for (int i = 0; i < first_conv_output_channels; i++)
	{

		first_conv_hidden_weights(i) = -initial_first_hidden_weigt_radius * cube(first_conv_input_channels, K, K, fill::ones) + 2.0 * initial_first_hidden_weigt_radius * cube(first_conv_input_channels, K, K, fill::randu);
	}
	// second convolutional hidden field
	for (int i = 0; i < second_conv_output_channels; i++)
	{

		second_conv_hidden_weights(i) = -initial_second_hidden_weigt_radius * cube(second_conv_input_channels, K, K, fill::ones) + 2.0 * initial_second_hidden_weigt_radius * cube(second_conv_input_channels, K, K, fill::randu);
	}

	first_hidden_weights_mat = unroll_kernel_to_weight_mat(first_conv_output_channels, first_conv_input_channels, first_conv_hidden_weights, M_root);
	second_hidden_weights_mat = unroll_kernel_to_weight_mat(second_conv_output_channels, second_conv_input_channels, second_conv_hidden_weights, M_root);
}

void get_targets(double (&targets)[P], int label)
{
	/*
	Function to convert label (e.g. 2) to target vector (e.g. (0, 0, 1, 0, ...)).

	Args:
	targets: reference to double array of length P = 10.
			 To be filled with 0.0 and 1.0, where the position of the 1.0 is determined by the label.
	label: int.
		   Number between 0 and 9.
	*/
	for (int p = 0; p < P; ++p)
	{
		if (p == label)
		{
			targets[p] = 1.0;
		}
		else
		{
			targets[p] = 0.0;
		}
	}
}

void print_parameters(string results_file_name, string print_msg, string task, string system_simu, double theta, double alpha,
					  int number_of_epochs, double eta_0, double eta_1,
					  int N_h, double exp_precision)
{
	/*
	Function to create text file in which the parameters and results will be saved.
	If a file with the same name already exists, it will be overridden.
	This functions prints the parameters and an initial message to the text file.
	The results will be appended later by the function "print_results".

	Args:
	results_file_name:	string.
						Name of the text file.
	print_msg:			string.
						Message to write at the beginning of the text file.
	all other:			diverse datatypes.
						Simulation options and parameters to be printed to the text file.
	*/

	// get current time and date
	auto current_clock = chrono::system_clock::now();
	time_t current_time = chrono::system_clock::to_time_t(current_clock);

	ofstream results_file;
	results_file.open(results_file_name);
	results_file << print_msg << endl;
	results_file << endl;
	results_file << "start of simulation: " << ctime(&current_time);
	results_file << endl;
	results_file << endl;
	results_file << "OPTIONS:" << endl;
	results_file << endl;
	results_file << "task: " << task << endl;
	results_file << "method to solve the DDE (or network): " << system_simu << endl;
	//	results_file << "method to compute the gradient: " << grad_comp << endl;
	results_file << endl;
	results_file << endl;
	results_file << "PARAMETERS:" << endl;
	results_file << endl;
	results_file << "System Parameters:" << endl;
	results_file << "theta = " << theta << endl;
	results_file << "alpha = " << alpha << endl;
	results_file << endl;
	results_file << "Training Parameters:" << endl;
	results_file << "number_of_epochs = " << number_of_epochs << endl;
	results_file << "eta_0 = " << eta_0 << endl;
	results_file << "eta_1 = " << eta_1 << endl;
	results_file << endl;
	results_file << "RESULTS:" << endl;
	results_file.close();
}

void print_results(string results_file_name, vector<double> training_accuracy_vector, vector<double> accuracy_vector)
{
	/*
	Function to append results (i.e. training and validation accuracies for each epoch) to the results text file.
	Moreover the index of the validation batch and the diagonal indices n'_d are printed to the file.

	Args:
	results_file_name:			string.
								Name of the text file.
	training_accuracy_vector:	vector<double>.
								Vector containing the training accuracies for each epoch.
	accuracy_vector:			vector<double>.
								Vector containing the validation accuracies for each epoch.
	*/
	int epochs = accuracy_vector.size();

	ofstream results_file;
	results_file.open(results_file_name, ios_base::app);
	results_file << endl;
	results_file << "Results for test set" << endl;

	for (int e = 0; e < epochs; ++e)
	{
		stringstream stream_train;
		stream_train << fixed << setprecision(6) << training_accuracy_vector[e];
		string s_train = stream_train.str();

		stringstream stream_val;
		stream_val << fixed << setprecision(6) << accuracy_vector[e];
		string s_val = stream_val.str();

		// Write the accuracy values to the file
		results_file << "Epoch " << e + 1 << ": Training Accuracy = " << s_train << ", Test Accuracy = " << s_val << endl;
	}
	results_file.close();
}
