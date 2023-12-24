#include "solve_network.h"

using namespace std;
using namespace arma;
using namespace globalconstants;

void solve_network(vec &activations_first_hidden_layer, vec &activations_second_hidden_layer, vec &activations_third_hidden_layer, vec &node_states_first_hidden_layer, vec &node_states_second_hidden_layer, vec &node_states_third_hidden_layer, double (&output_activations)[P], double (&outputs)[P], vec &g_primes,
				   vec &input_data, vec input_weights, mat first_hidden_weights_mat, mat second_hidden_weights_mat, mat output_weights, double theta, double alpha,
				   std::vector<int> first_conv_diag_indices, std::vector<int> second_conv_diag_indices)
{
	/*
	Function to solve the network equations.
	Provides states of the hidden and output nodes and their activations.

	Args:

	activations_first_hidden_layer:   reference to arma:vec with size M = #first_conv_input_channels * M_root * M_root
									  To be filled with activations of the first hidden layer nodes
	activations_second_hidden_layer:   reference to arma:vec with size #first_conv_output_channels * M_root * M_root
									  To be filled with activations of the second hidden layer nodes
	activations_third_hidden_layer:   reference to arma:vec with size #second_conv_output_channels * M_root * M_root
									  To be filled with activations of the third hidden layer nodes

	node_states_first_hidden_layer:   reference to arma:vec with size M = #first_conv_input_channels * M_root * M_root
									  To be filled with states of the first hidden layer nodes
	node_states_second_hidden_layer:   reference to arma:vec with size #first_conv_output_channels * M_root * M_root
									  To be filled with states of the second hidden layer nodes
	node_states_third_hidden_layer:   reference to arma:vec with size #second_conv_output_channels * M_root * M_root
									  To be filled with states of the third hidden layer nodes

	output_activations: reference to double array of size P = 10.
						To be filled with the activations of the output nodes.

	outputs:            reference to double array of size P = 10.
						To be filled with the output node states.

	input_data:         reference to arma::vec of length M.
						Input vector. Contains the pixel values of an input image.

	input_weights:      reference to arma::mat of size M x (M + 1)
						Matrix W^in. Contains the weights connecting the input layer

	first_hidden_weights_mat: reference to arma::mat of size #first_conv_output_channels * M_root * M_root
						Matrix containting the weights of the connecting layer 2 to layer 1

	second_hidden_weights_mat: reference to arma::mat of size #second_conv_output_channels * M_root * M_root
						Matrix containting the weights of the connecting layer 3 to layer 2

	output_weights:     reference to arma::mat with size P x (#second_cond_output_channels * M_root * M_root + 1) (where P = 10).
						Contains the weights connecting the last hidden layer to the output layer.

	theta:              double.
						Node Separation. theta = T / N.

	alpha:              double.
						Factor for the linear dynamics. Must be negative.
	*/

	double x_0 = 0.0;

	double exp_factor = exp(alpha * theta);
	double phi = (exp_factor - 1.0) / alpha;

	double summe = 0.0;

	// We compute the activations of the first layer:
	for (int n = 0; n < M; ++n)
	{
		summe = 0;
		summe += input_weights(n) * input_data(n);
	
		g_primes(n) = input_processing_prime(summe);
		activations_first_hidden_layer(n) = input_processing(summe);
	}

	// compute node states for first hidden layer
	node_states_first_hidden_layer(0) = exp_factor * x_0 + phi * f(activations_first_hidden_layer(0));
	for (int n = 1; n < M; ++n)
	{
		node_states_first_hidden_layer(n) = exp_factor * node_states_first_hidden_layer(n - 1) + phi * f(activations_first_hidden_layer(n));
	}

	// compute activations and node states for the second layer
	for (int n = 0; n < first_conv_input_channels * M_root * M_root; ++n)
	{
		summe = 0.0;
		for (int n_prime_d : first_conv_diag_indices)
		{
			int j = n - n_prime_d;
			if (j >= M)
			{
				continue;
			}
			if (j < 0)
			{
				break;
			}
			summe += first_hidden_weights_mat(n, j) * node_states_first_hidden_layer(j);
		}
		activations_second_hidden_layer(n) = summe;
	}

	node_states_second_hidden_layer(0) = exp_factor * node_states_first_hidden_layer(M - 1) + phi * f(activations_second_hidden_layer(0));
	for (int n = 1; n < first_conv_output_channels * M; ++n)
	{
		node_states_second_hidden_layer(n) = exp_factor * node_states_second_hidden_layer(n - 1) + phi * f(activations_second_hidden_layer(n));
	}

	// compute the node states for third hidden layer
	for (int n = 0; n < second_conv_output_channels * M_root * M_root; ++n)
	{
		summe = 0.0;
		for (int n_prime_d : second_conv_diag_indices)
		{
			int j = n - n_prime_d;
			if (j >= first_conv_output_channels * M_root * M_root)
			{
				continue;
			}
			if (j < 0)
			{
				break;
			}
			summe += second_hidden_weights_mat(n, j) * node_states_second_hidden_layer(j);
		}
		activations_third_hidden_layer(n) = summe;
		if (skip == 1)
		{
			activations_third_hidden_layer(n) = summe + node_states_second_hidden_layer(n);
		}
	}

	node_states_third_hidden_layer(0) = exp_factor * node_states_second_hidden_layer(first_conv_output_channels * M_root * M_root - 1) + phi * f(activations_third_hidden_layer(0));
	for (int n = 1; n < second_conv_output_channels * M_root * M_root; ++n)
	{
		node_states_third_hidden_layer(n) = exp_factor * node_states_third_hidden_layer(n - 1) + phi * f(activations_third_hidden_layer(n));
	}

	// compute output activations
	for (int p = 0; p < P; ++p)
	{
		summe = 0.0;
		summe = output_weights(p, second_conv_output_channels * M_root * M_root); // bias weight
		for (int n = 0; n < second_conv_output_channels * M_root * M_root; ++n)
		{
			summe += output_weights(p, n) * node_states_third_hidden_layer(n);
		}
		output_activations[p] = summe;
	}

	// compute outputs with softmax function
	double exp_sum = 0;
	for (int p = 0; p < P; ++p)
	{
		exp_sum += exp(output_activations[p]);
	}
	for (int p = 0; p < P; ++p)
	{
		outputs[p] = exp(output_activations[p]) / exp_sum;
	}
}
