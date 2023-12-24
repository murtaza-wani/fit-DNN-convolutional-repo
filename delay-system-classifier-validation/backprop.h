#ifndef BACKPROP_H
#define BACKPROP_H

#include <iostream>
#include <armadillo>

#include "global_constants.h"

void get_gradient_node_by_node(arma::vec &input_weight_gradient, arma::field<arma::cube> &first_conv_weight_gradient, arma::field<arma::cube> &second_conv_weight_gradient, arma::mat &output_weight_gradient, arma::vec &input_data, arma::vec &node_states_first_hidden_layer, arma::vec &node_states_second_hidden_layer, arma::vec &node_states_third_hidden_layer, arma::vec &f_prime_activations_first_hidden_layer, arma::vec &f_prime_activations_second_hidden_layer, arma::vec &f_prime_activations_third_hidden_layer, arma::vec &g_primes, double (&outputs)[globalconstants::P], double (&targets)[globalconstants::P], arma::mat first_hidden_weights_mat, arma::mat second_hidden_weights_mat, arma::mat output_weights, std::vector<int> first_conv_diag_indices, std::vector<int> second_conv_diag_indices, double theta, double alpha);

#endif
