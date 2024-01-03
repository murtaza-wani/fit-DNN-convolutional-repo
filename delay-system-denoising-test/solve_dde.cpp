#include "solve_dde.h"

using namespace std;
using namespace arma;
using namespace globalconstants;

void solve_dde_heun(vec &activations_first_hidden_layer, vec &activations_second_hidden_layer, vec &activations_third_hidden_layer, vec &node_states_first_hidden_layer, vec &node_states_second_hidden_layer, vec &node_states_third_hidden_layer, double (&output_activations)[P], double (&outputs)[P], vec &g_primes,
					vec &input_data, vec input_weights, mat first_hidden_weights_mat, mat second_hidden_weights_mat, mat output_weights, vector<int> &first_conv_diag_indices, vector<int> &second_conv_diag_indices, double theta, double alpha,
					int N_h)
{
	/*
	Function to solve the delay differential equations by a semi-analytic Heun method.
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

	input_data:         reference to arma::vec of length M= #first_conv_input_channels * M_root * M_root
						Input vector. Contains the pixel values of an input image.

	input_weights:      reference to arma::mat of size M 
						Vec W^in. Contains the weights connecting the input layer to the first hidden layer

	first_hidden_weights_mat: reference to arma::mat of size #first_conv_output_channels * M_root * M_root
						Matrix containting the weights of the connecting layer 2 to layer 1

	second_hidden_weights_mat: reference to arma::mat of size #second_conv_output_channels * M_root * M_root
						Matrix containting the weights of the connecting layer 3 to layer 2

	output_weights:     reference to arma::mat with size P x (#second_cond_output_channels * M_root * M_root + 1) (where P = 10).
						Contains the weights connecting the last hidden layer to the output layer.

	first_conv_diagonal_indices: reference to arma::vec
								Contains indices of all non-zero diagonals in first_hidden_weights_mat

	second_conv_diagonal_indices: reference to arma::vec
								Contains indices of all non-zero diagonals in second_hidden_weights_mat


	theta:              double.
						Node Separation. theta = T / N.

	alpha:              double.
						Factor for the linear dynamics. Must be negative.
	*/

	double x_0 = 0.0;
	double summe;
	double h = theta / (double)N_h;
	double exp_factor_h = exp(alpha * h);

	// In the following we often need the values exp(alpha * n_h * h). That's why we make a lookup table.
	vector<double> exp_h_table;
	for (int n_h = 0; n_h < N_h + 1; ++n_h)
	{
		double exp_h_value = exp(alpha * h * (double)n_h);
		exp_h_table.push_back(exp_h_value);
	}

	// We need to memorize the values of and a^\ell_{n, n_h} and x^\ell_{n, n_h}, one layer at at a time
	// We also need a matrix to save the activated states f( a^\ell_{n, n_h} ) to accelerate the computation.
	// Thus, we use armadillo matrices:

	mat a_states_first_hidden_layer(first_conv_input_channels * M_root * M_root, N_h + 1);
	mat fa_states_first_hidden_layer(first_conv_input_channels * M_root * M_root, N_h + 1);
	mat x_states_first_hidden_layer(first_conv_input_channels * M_root * M_root, N_h);

	mat a_states_second_hidden_layer(first_conv_output_channels * M_root * M_root, N_h + 1);
	mat fa_states_second_hidden_layer(first_conv_output_channels * M_root * M_root, N_h + 1);
	mat x_states_second_hidden_layer(first_conv_output_channels * M_root * M_root, N_h);

	mat a_states_third_hidden_layer(second_conv_output_channels * M_root * M_root, N_h + 1);
	mat fa_states_third_hidden_layer(second_conv_output_channels * M_root * M_root, N_h + 1);
	mat x_states_third_hidden_layer(second_conv_output_channels * M_root * M_root, N_h);

	// We compute the activations of the first layer:
	for (int n = 0; n < M; ++n)
	{
		summe = 0;
		summe += input_weights(n) * input_data(n);
		
		g_primes(n) = input_processing_prime(summe);
		activations_first_hidden_layer(n) = input_processing(summe);
	}

	// For the first layer, we can calculate the exact values of x.

	// We begin with the x values up to the first node:
	for (int n_h = 0; n_h < N_h; ++n_h)
	{
		x_states_first_hidden_layer(0, n_h) = exp_h_table[n_h + 1] * x_0 + 1.0 / alpha * (exp_h_table[n_h + 1] - 1.0) * f(activations_first_hidden_layer(0));
	}
	// Now we know the value x(theta) of the first node of the first layer:
	node_states_first_hidden_layer(0) = x_states_first_hidden_layer(0, N_h - 1);
	// We continue with the calculation of the remaining x values for the first layer:
	for (int n = 1; n < M; ++n)
	{
		for (int n_h = 0; n_h < N_h; ++n_h)
		{
			x_states_first_hidden_layer(n, n_h) = exp_h_table[n_h + 1] * x_states_first_hidden_layer(n - 1, N_h - 1) + 1.0 / alpha * (exp_h_table[n_h + 1] - 1.0) * f(activations_first_hidden_layer(n));
		}
		// From the x_states of the first layer, we save the values of the node states:
		node_states_first_hidden_layer(n) = x_states_first_hidden_layer(n, N_h - 1);
	}

	// For second hidden layer

	// first we get the a_states for the current layer:
	for (int n = 0; n < first_conv_output_channels * M_root * M_root; ++n)
	{
		// a_states for n_h = 0:

		summe = first_hidden_weights_mat(n, 0) * x_0;

		// other summands:
		for (int n_prime_d : first_conv_diag_indices)
		{
			int j = n - n_prime_d;
			if (j >= first_conv_input_channels * M_root * M_root)
			{
				continue;
			}
			if (j < 1)
			{
				break;
			}
			summe += first_hidden_weights_mat(n, j) * node_states_first_hidden_layer(j - 1);
		}
		a_states_second_hidden_layer(n, 0) = summe;
		// a_states for other n_h:
		for (int n_h = 0; n_h < N_h; ++n_h)
		{
			summe = 0;
			for (int n_prime_d : first_conv_diag_indices)
			{
				int j = n - n_prime_d;
				if (j >= first_conv_input_channels * M_root * M_root)
				{
					continue;
				}
				if (j < 0)
				{
					break;
				}
				summe += first_hidden_weights_mat(n, j) * x_states_first_hidden_layer(j, n_h);
			}
			a_states_second_hidden_layer(n, n_h + 1) = summe;
		}
		// the last a_state on a theta-interval is the activation:
		activations_second_hidden_layer(n) = a_states_second_hidden_layer(n, N_h);
	}
	// get f(a):
	fa_states_second_hidden_layer = f_matrix(a_states_second_hidden_layer);
	// Now we can compute the x_states for the current layer:
	// first case: n = 1 and n_h = 1:
	x_states_second_hidden_layer(0, 0) = exp_factor_h * node_states_first_hidden_layer(M - 1) + 0.5 * h * (exp_factor_h * fa_states_second_hidden_layer(0, 0) + fa_states_second_hidden_layer(0, 1));
	// second case: n = 1 and n_h > 1:
	for (int n_h = 1; n_h < N_h; ++n_h)
	{
		x_states_second_hidden_layer(0, n_h) = exp_factor_h * x_states_second_hidden_layer(0, n_h - 1) + 0.5 * h * (exp_factor_h * fa_states_second_hidden_layer(0, n_h) + fa_states_second_hidden_layer(0, n_h + 1));
	}
	// node state is x_state on theta-grid-point:
	node_states_second_hidden_layer(0) = x_states_second_hidden_layer(0, N_h - 1);
	for (int n = 1; n < first_conv_output_channels * M_root * M_root; ++n)
	{
		// third case: n > 1 and n_h = 1:
		x_states_second_hidden_layer(n, 0) = exp_factor_h * x_states_second_hidden_layer(n - 1, N_h - 1) + 0.5 * h * (exp_factor_h * fa_states_second_hidden_layer(n, 0) + fa_states_second_hidden_layer(n, 1));
		// fourth case: n > 1 and n_h > 1:
		for (int n_h = 1; n_h < N_h; ++n_h)
		{
			x_states_second_hidden_layer(n, n_h) = exp_factor_h * x_states_second_hidden_layer(n, n_h - 1) + 0.5 * h * (exp_factor_h * fa_states_second_hidden_layer(n, n_h) + fa_states_second_hidden_layer(n, n_h + 1));
		}
		// node state is x_state on theta-grid-point:
		node_states_second_hidden_layer(n) = x_states_second_hidden_layer(n, N_h - 1);
	}

	// For third hidden layer

	for (int n = 0; n < second_conv_output_channels * M_root * M_root; ++n)
	{
		// a_states for n_h = 0:
		summe = second_hidden_weights_mat(n, 0) * node_states_first_hidden_layer(M - 1);

		// other summands:
		for (int n_prime_d : second_conv_diag_indices)
		{
			int j = n - n_prime_d;
			if (j >= first_conv_output_channels * M_root * M_root)
			{
				continue;
			}
			if (j < 1)
			{
				break;
			}
			summe += second_hidden_weights_mat(n, j) * node_states_second_hidden_layer(j - 1);
		}
		a_states_third_hidden_layer(n, 0) = summe;

		// a_states for other n_h:
		for (int n_h = 0; n_h < N_h; ++n_h)
		{
			summe = 0;
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
				summe += second_hidden_weights_mat(n, j) * x_states_second_hidden_layer(j, n_h);
			}
			a_states_third_hidden_layer(n, n_h + 1) = summe;
		}
		// the last a_state on a theta-interval is the activation:
		activations_third_hidden_layer(n) = a_states_third_hidden_layer(n, N_h);
	}
	// get f(a):
	fa_states_third_hidden_layer = f_matrix(a_states_third_hidden_layer);
	// Now we can compute the x_states for the current layer:
	// first case: n = 1 and n_h = 1:
	x_states_third_hidden_layer(0, 0) = exp_factor_h * node_states_second_hidden_layer(first_conv_output_channels * M_root * M_root - 1) + 0.5 * h * (exp_factor_h * fa_states_third_hidden_layer(0, 0) + fa_states_third_hidden_layer(0, 1));
	// second case: n = 1 and n_h > 1:
	for (int n_h = 1; n_h < N_h; ++n_h)
	{
		x_states_third_hidden_layer(0, n_h) = exp_factor_h * x_states_third_hidden_layer(0, n_h - 1) + 0.5 * h * (exp_factor_h * fa_states_third_hidden_layer(0, n_h) + fa_states_third_hidden_layer(0, n_h + 1));
	}
	// node state is x_state on theta-grid-point:
	node_states_third_hidden_layer(0) = x_states_third_hidden_layer(0, N_h - 1);
	for (int n = 1; n < second_conv_output_channels * M_root * M_root; ++n)
	{
		// third case: n > 1 and n_h = 1:
		x_states_third_hidden_layer(n, 0) = exp_factor_h * x_states_third_hidden_layer(n - 1, N_h - 1) + 0.5 * h * (exp_factor_h * fa_states_third_hidden_layer(n, 0) + fa_states_third_hidden_layer(n, 1));
		// fourth case: n > 1 and n_h > 1:
		for (int n_h = 1; n_h < N_h; ++n_h)
		{
			x_states_third_hidden_layer(n, n_h) = exp_factor_h * x_states_third_hidden_layer(n, n_h - 1) + 0.5 * h * (exp_factor_h * fa_states_third_hidden_layer(n, n_h) + fa_states_third_hidden_layer(n, n_h + 1));
		}
		// node state is x_state on theta-grid-point:
		node_states_third_hidden_layer(n) = x_states_third_hidden_layer(n, N_h - 1);
	}

	// compute output activations
	for (int p = 0; p < P; ++p)
	{
		summe = output_weights(p, second_conv_output_channels * M_root * M_root);  // bias weight
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
