#ifndef SOLVE_DDE_H
#define SOLVE_DDE_H

#include <iostream>
#include <armadillo>

#include "f.h"
#include "global_constants.h"

void solve_dde_heun(arma::vec &activations_first_hidden_layer, arma::vec &activations_second_hidden_layer, arma::vec &activations_third_hidden_layer, arma::vec &node_states_first_hidden_layer, arma::vec &node_states_second_hidden_layer, arma::vec &node_states_third_hidden_layer, double (&output_activations)[globalconstants::P], double (&outputs)[globalconstants::P], arma::vec &g_primes,
					arma::vec &input_data, arma::vec input_weights, arma::mat first_hidden_weights_mat, arma::mat second_hidden_weights_mat, arma::mat output_weights, std::vector<int> &first_conv_diag_indices, std::vector<int> &second_conv_diag_indices, double theta, double alpha,
					int N_h);

#endif
