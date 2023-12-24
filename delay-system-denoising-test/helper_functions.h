#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <armadillo>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <ctime>

#include "global_constants.h"

extern std::mt19937 rng;
double uniform(double a, double b);
double learning_rate(int epoch, int k, double eta_0, double eta_1);
void read_files(arma::cube &train_images, arma::mat &test_images, int (&train_labels)[globalconstants::number_of_training_batches][globalconstants::training_batch_size], int (&test_labels)[globalconstants::test_batch_size], std::string data_dir);
void initialize_weights(arma::vec &input_weights, arma::field<arma::cube> &first_conv_hidden_weights, arma::field<arma::cube> &second_conv_hidden_weights, arma::mat &first_hidden_weights_mat, arma::mat &second_hidden_weights_mat, arma::mat &output_weights, double initial_input_weigt_radius, double initial_first_hidden_weigt_radius, double initial_second_hidden_weigt_radius, double initial_output_weigt_radius, bool save_to_file);
void get_targets(double (&targets)[globalconstants::P], int label);
void print_parameters(std::string results_file_name, std::string print_msg, std::string task, std::string system_simu, double theta, double alpha, int number_of_epochs, double eta_0, double eta_1, int N_h, double exp_precision);
void print_results(std::string results_file_name, std::vector<double> training_accuracy_vector, std::vector<double> accuracy_vector);
arma::Mat<int> get_kernel_weight_locations(int t, int s, int j, int l, int K, int M_root); //(t,s,j,l) represents a convolutional weight
arma::mat unroll_kernel_to_weight_mat(int output_channels, int input_channels, const arma::field<arma::cube> &kernels, int M_root);
std::vector<int> diagonal_indices(arma::mat &unrolled);

#endif
