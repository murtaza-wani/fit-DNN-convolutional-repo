#ifndef SOLVE_NETWORK_H
#define SOLVE_NETWORK_H

#include <iostream>
#include <armadillo>

#include "f.h"
#include "global_constants.h"

void solve_network(arma::vec &activations_first_hidden_layer, arma::vec &activations_second_hidden_layer, arma::vec &activations_third_hidden_layer, arma::vec &node_states_first_hidden_layer, arma::vec &node_states_second_hidden_layer, arma::vec &node_states_third_hidden_layer, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes,
				   arma::vec &input_data, arma::vec input_weights, arma::mat first_hidden_weights_mat, arma::mat second_hidden_weights_mat, arma::mat output_weights, double theta, double alpha,
				   std::vector<int> first_conv_diag_indices, std::vector<int> second_conv_diag_indices);

#endif
