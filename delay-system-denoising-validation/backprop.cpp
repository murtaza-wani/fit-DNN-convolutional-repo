#include "backprop.h"
#include "helper_functions.h"
#include "global_constants.h"

using namespace std;
using namespace arma;
using namespace globalconstants;

void get_gradient_node_by_node(vec &input_weight_gradient, field<cube> &first_conv_weight_gradient, field<cube> &second_conv_weight_gradient, mat &output_weight_gradient, vec &input_data, vec &node_states_first_hidden_layer, vec &node_states_second_hidden_layer, vec &node_states_third_hidden_layer, vec &f_prime_activations_first_hidden_layer, vec &f_prime_activations_second_hidden_layer,
							   vec &f_prime_activations_third_hidden_layer, vec &g_primes, double (&outputs)[P], double (&targets)[P], mat first_hidden_weights_mat, mat second_hidden_weights_mat, mat output_weights, vector<int> first_conv_diag_indices, vector<int> second_conv_diag_indices, double theta, double alpha)
{
	/*
	Function to compute the gradient using node-by-node backpropagation.
	For the hidden weights only the gradients for the nonzero diagonals (and the bias weights) must be nonzero.

	Args:
	input_weight_gradient: reference to arma::vec of size M.
						   To be filled with partial derivatives w.r.t. input weights.
	first_conv_weight_gradient: reference to arma:field of size #first_conv_output_channels * #first_conv_input_channels * K * K
						   To be filled with partial derivatives w.r.t. first covolutional weights
	second_conv_weight_gradient: reference to arma:field of size #second_conv_output_channels * #second_conv_input_channels * K * K
						   To be filled with partial derivatives w.r.t. second covolutional weights

	output_weight_gradient: reference to arma::mat of size P x (#second_conv_output_channels * M_root * M_root + 1).
							To be filled with partial derivatives w.r.t. output weights.
	input_data: reference to arma::vec of length M = #first_conv_input_channels * M_root * M_root
				Input vector. Contains the pixel values of an input image.
	node_states_first_hidden_layer: reference to arma::vec with size M
				 Contains states of the nodes of first hidden layer.
	node_states_second_hidden_layer: reference to arma::vec with size #first_conv_output_channels * M_root * M_root
				 Contains states of the nodes of second hidden layer.
	node_states_third_hidden_layer: reference to arma::vec with size #second_conv_output_channels * M_root * M_root
				 Contains states of the nodes of third hidden layer.
	f_prime_activations_first_hidden_layers: reference to arma:vec with size M
						 Contains the values of f' at the activations of the system (or network) for the first hidden layer.
	f_prime_activations_second_hidden_layers: reference to arma:vec with size #first_conv_output_channels * M_root * M_root
						 Contains the values of f' at the activations of the system (or network) for the second hidden layer.
	f_prime_activations_third_hidden_layers: reference to arma:vec with size #second_conv_output_channels * M_root * M_root
						 Contains the values of f' at the activations of the system (or network) for the third hidden layer.

	g_primes: reference to armadillo vector of size M:
			  Contains the values of g'(a^in_n).
	outputs: reference to double array of length P = 10.
			 Contains the outputs of the system.
	targets: reference to double array of length P = 10.
			 Contains the target which should be matched by the outputs.
	first_hidden_weights_mat: reference to armadillo mat of size (#first_conv_output_channels * M_root * M_root) x (#first_conv_input_channels * M_root * M_root)
							contains the hidden weights from first conv hidden weights
	second_hidden_weights_mat: reference to armadillo mat of size (#second_conv_output_channels * M_root * M_root) x (#second_conv_input_channels * M_root * M_root)
							contains the hidden weights from first conv hidden weights

	output_weights: reference to armadillo matrix of size P x (#second_conv_output_channels * M_root * M_root + 1), where P = 10.
					Contains the output weights.
	theta: double.
		   Node Separation. theta = T / N.
	alpha: double.
		   Factor for the linear dynamics. Must be negative.
	*/

	// arrays to store partial derivatives:
	vec deltas_first_hidden_layer(first_conv_input_channels * M_root * M_root);
	deltas_first_hidden_layer.zeros();
	vec deltas_second_hidden_layer(first_conv_output_channels * M_root * M_root);
	deltas_second_hidden_layer.zeros();
	vec deltas_third_hidden_layer(second_conv_output_channels * M_root * M_root);
	deltas_third_hidden_layer.zeros();
	
	double Delta;
	double output_deltas[P];

	double local_coupling = exp(alpha * theta);
	double phi = (local_coupling - 1.0) / alpha;

	// Step (i)
	// get output deltas (partial derivatives w.r.t. output activations)
	for (int p = 0; p < P; ++p)
	{
		output_deltas[p] = outputs[p] - targets[p];
	}

	// Step (ii)

	// get Deltas for last(third) hidden layer (partial derivatives w.r.t. node states)
	// and get deltas for last(third) hidden layer (partial derivatives w.r.t. node activations)
	Delta = 0;
	for (int p = 0; p < P; ++p)
	{
		Delta += output_deltas[p] * output_weights(p, second_conv_output_channels * M_root * M_root - 1);
	}
	deltas_third_hidden_layer(second_conv_output_channels * M_root * M_root - 1) = Delta * phi * f_prime_activations_third_hidden_layer(second_conv_output_channels * M_root * M_root - 1);
	for (int n = second_conv_output_channels * M_root * M_root - 2; n >= 0; --n)
	{
		Delta = local_coupling * Delta;
		for (int p = 0; p < P; ++p)
		{
			Delta += output_deltas[p] * output_weights(p, n);
		}
		deltas_third_hidden_layer(n) = Delta * phi * f_prime_activations_third_hidden_layer(n);
	}

	// get deltas for second hidden layer
	Delta = local_coupling * Delta;
	for (int n_prime_d : second_conv_diag_indices)
	{
		int i = first_conv_output_channels * M_root * M_root - 1 + n_prime_d;
		if (i < 0)
		{
			continue;
		}
		if (i >= second_conv_output_channels * M_root * M_root)
		{

			break;
		}
		Delta += deltas_third_hidden_layer(i) * second_hidden_weights_mat(i, first_conv_output_channels * M_root * M_root - 1);
	}

	deltas_second_hidden_layer(first_conv_output_channels * M_root * M_root - 1) = Delta * phi * f_prime_activations_second_hidden_layer(first_conv_output_channels * M_root * M_root - 1);
	for (int n = first_conv_output_channels * M_root * M_root - 2; n >= 0; --n)
	{
		Delta = local_coupling * Delta;
		for (int n_prime_d : second_conv_diag_indices)
		{
			int i = n + n_prime_d;
			if (i < 0)
			{
				continue;
			}
			if (i >= second_conv_output_channels * M_root * M_root)
			{
				break;
			}
			Delta += deltas_third_hidden_layer(i) * second_hidden_weights_mat(i, n);
		}

		deltas_second_hidden_layer(n) = Delta * phi * f_prime_activations_second_hidden_layer(n);
	}

	// get deltas for 1st hidden layer
	Delta = local_coupling * Delta;
	for (int n_prime_d : first_conv_diag_indices)
	{
		int i = first_conv_input_channels * M_root * M_root - 1 + n_prime_d;
		if (i < 0)
		{
			continue;
		}
		if (i >= first_conv_output_channels * M_root * M_root)
		{

			break;
		}
		Delta += deltas_second_hidden_layer(i) * first_hidden_weights_mat(i, first_conv_input_channels * M_root * M_root - 1);
	}
	deltas_first_hidden_layer(first_conv_input_channels * M_root * M_root - 1) = Delta * phi * f_prime_activations_first_hidden_layer(first_conv_input_channels * M_root * M_root - 1);
	for (int n = first_conv_input_channels * M_root * M_root - 2; n >= 0; --n)
	{
		Delta = local_coupling * Delta;
		for (int n_prime_d : first_conv_diag_indices)
		{
			int i = n + n_prime_d;
			if (i < 0)
			{
				continue;
			}
			if (i >= first_conv_output_channels * M_root * M_root)
			{
				break;
			}
			Delta += deltas_second_hidden_layer(i) * first_hidden_weights_mat(i, n);
		}
		deltas_first_hidden_layer(n) = Delta * phi * f_prime_activations_first_hidden_layer(n);
	}

	// Steps (iv)-(vi)

	// input weight gradient
	for (int n = 0; n < M; ++n)
	{
		
			input_weight_gradient(n) = deltas_first_hidden_layer(n) * input_data(n) * g_primes(n);
		
	}

	// second conv weight gradient
	for (int t = 0; t < second_conv_output_channels; t++)
	{
		for (int s = 0; s < second_conv_input_channels; s++)
		{
			for (int n = 0; n < K; ++n)
			{
				for (int j = 0; j < K; ++j)
				{
					Mat<int> arr = get_kernel_weight_locations(t, s, n, j, K, M_root);
					int counter = arr.n_rows;
					double sum = 0.0;
					for (int i = 0; i < counter; ++i)
					{
						sum += deltas_third_hidden_layer(arr(i, 0)) * node_states_second_hidden_layer(arr(i, 1));
					}

					second_conv_weight_gradient(t)(s, n, j) = sum;
				}
			}
		}
	}

	// first conv weight gradient

	for (int t = 0; t < first_conv_output_channels; t++)
	{
		for (int s = 0; s < first_conv_input_channels; s++)
		{
			for (int n = 0; n < K; ++n)
			{
				for (int j = 0; j < K; ++j)
				{
					Mat<int> arr = get_kernel_weight_locations(t, s, n, j, K, M_root);
					int counter = arr.n_rows;
					double sum = 0.0;
					for (int i = 0; i < counter; ++i)
					{
						sum += deltas_second_hidden_layer(arr(i, 0)) * node_states_first_hidden_layer(arr(i, 1));
					}

					first_conv_weight_gradient(t)(s, n, j) = sum;
				}
			}
		}
	}

	// output weight gradient
	for (int p = 0; p < P; ++p)
	{
		for (int n = 0; n < second_conv_output_channels * M_root * M_root; ++n)
		{
			output_weight_gradient(p, n) = output_deltas[p] * node_states_third_hidden_layer(n);
		}
		output_weight_gradient(p, second_conv_output_channels * M_root * M_root) = output_deltas[p]; // for bias output weight
	}
}
