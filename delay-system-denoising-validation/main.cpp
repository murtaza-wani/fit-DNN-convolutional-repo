#include <iostream>
#include <armadillo>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include "helper_functions.h"
#include "solve_network.h"
#include "solve_dde.h"
#include "backprop.h"
#include "katana_get_params.hh"
#include "global_constants.h"

using namespace std;
using namespace arma;
using namespace globalconstants;

int main(int argc, char const *argv[])

{
    // for cpu time measuring
    clock_t start_overall = clock();
    double cumulative_solve_time = 0.0;
    double cumulative_backprop_time = 0.0;

    arma_rng::set_seed_random();

    // ### ### ### --- OPTIONS --- ### ### ###

    // file name of the text file where parameters and results will be saved
    string results_file_name = katana::getCmdOption(argv, argv + argc, "-filename", "results.txt");
    // print message to results file
    string print_msg = "Simulation of the deep learning delay system.";

    // task, possible options are "MNIST", "Fashion-MNIST", "CIFAR-10", "SVHN"
    string task = katana::getCmdOption(argv, argv + argc, "-task", "Fashion-MNIST-denoising");
    // modify global_constants.h accordingly!

    // number of example images to save
    int save_examples = katana::getCmdOption(argv, argv + argc, "-save_examples", 0);

    // print weights (and diagonals) after each training epoch to text files in weights folder
    bool print_weights_to_file = false;

    // if between 1 and 6, then cross_validation will be set to false the simulation will only be done for this validation batch
    int validation_batch = katana::getCmdOption(argv, argv + argc, "-validation_batch", 1);
    // if true: 6-fold cross validation
    // if false: validation using the given batch, training with the remaining batches
    bool cross_validation = true;
    if (validation_batch > 0)
    {
        cross_validation = false;
    }

    // option for system simulation
    // "dde_heun":	semi-analytic heun method with standard trapezoidal rule
    // "network":	solves the equivalent network equations
    string system_simu = katana::getCmdOption(argv, argv + argc, "-system_simu", "dde_heun");

    // option for gradient computation
    // backprop_standard: newly derived backpropagation algorithm for the deep learning delay system (was derived for equivalent network)
    string grad_comp = katana::getCmdOption(argv, argv + argc, "-grad_comp", "backprop_standard");

    // ### ### ### --- PARAMETERS --- ### ### ###

    // M and P are defined in global_constants.h

    // task parameter
    double noise_sigma = 1.0;

    // ... for the system/network architectur:

    double theta = katana::getCmdOption(argv, argv + argc, "-theta", 0.5); // node separation
    double alpha = -1.0;                                                   // factor in linear part of delay system

    // ... for the training:
    int number_of_epochs = katana::getCmdOption(argv, argv + argc, "-number_of_epochs", 1);
    double eta_0 = katana::getCmdOption(argv, argv + argc, "-eta0", 0.001);
    double eta_1 = katana::getCmdOption(argv, argv + argc, "-eta1", 1000.0); // learning rate eta = min(eta_0, eta_1 / step)
    

    // dropout
    double dropout_rate = katana::getCmdOption(argv, argv + argc, "-dropout", 0.0);

    // ... for weight initialization:
    double initial_input_weigt_radius = sqrt(2.0 / ((double)M + 1.0));
    double initial_first_hidden_weigt_radius = sqrt(2.0 / ((double)first_conv_input_channels * K * K + (double)first_conv_output_channels));
    double initial_second_hidden_weigt_radius = sqrt(2.0 / ((double)second_conv_input_channels * K * K + (double)second_conv_output_channels));
    double initial_output_weigt_radius = sqrt(2.0 / ((double)second_conv_output_channels * (double)M_root * (double)M_root + (double)P + 1.0));

    // ... for numerics:

    int N_h = max(32, (int)(16 * theta)); // computation steps per virtual node for solving DDE

    // computational precision for sums with exp(alpha * theta * n) factor in summands
    // in the functions "get_deltas" and "get_gradient" in "backprop.cpp".
    // exp_precision = -35.0 means that terms, where exponential factor
    // is smaller than exp(-35), are ignored.
    // Since exp(-35) is approximately 6.3e-16, the gradient will still be computed with double precision.
    double exp_precision = -35.0;

    // The following lines are to create a look up table "exp_table"
    // which contains the values exp(alpha * theta * n)
    // which are often needed by the functions "get_deltas" and "get_gradient" in "backprop.cpp".
    vector<double> exp_table;
    int n = 0;
    while (alpha * theta * double(n) >= -35.0)
    {
        double exp_value = exp(alpha * theta * double(n));
        exp_table.push_back(exp_value);
        ++n;
    }

    // make results file and print information about parameters
    print_parameters(results_file_name, print_msg, task,
                     cross_validation, system_simu,
                     theta, alpha,
                     number_of_epochs, eta_0, eta_1,
                     N_h, exp_precision);

    // determine data directory
    string data_dir;
    if (task == "MNIST-denoising")
    {
        data_dir = "data-MNIST";
    }
    else if (task == "Fashion-MNIST-denoising")
    {
        data_dir = "data-Fashion-MNIST";
    }
    else if (task == "CIFAR-10-denoising")
    {
        data_dir = "data-CIFAR-10";
    }
    else if (task == "SVHN-denoising")
    {
        data_dir = "data-SVHN";
    }
    else
    {
        cout << task << " is not a valid task." << endl;
        abort();
    }

    // read image data from files to arrays:
    cube train_images(number_of_training_batches, training_batch_size, M);
    mat test_images(test_batch_size, M);
    int train_labels[number_of_training_batches][training_batch_size];
    int test_labels[test_batch_size];
    read_files(train_images, test_images, train_labels, test_labels, data_dir);
    // The test images are not used in this version of the program (validation version).

    // ### ### ### --- CROSS VALIDATION --- ### ### ###

    int validation_batch_start_index;
    int validation_batch_end_index;
    if (cross_validation)
    {
        validation_batch_start_index = 0;
        validation_batch_end_index = number_of_training_batches - 1;
    }
    else
    {
        validation_batch_start_index = validation_batch - 1;
        validation_batch_end_index = validation_batch - 1;
    }

    // cross validation loop
    for (int validation_batch_index = validation_batch_start_index; validation_batch_index < validation_batch_end_index + 1; ++validation_batch_index)
    {

        vector<int> training_batch_indices;
        cout << "Begin training with training batches: ";
        for (int i = 0; i < number_of_training_batches; ++i)
        {
            if (i != validation_batch_index)
            {
                training_batch_indices.push_back(i);
                cout << i + 1 << ", ";
            }
        }
        cout << "and validation batch: " << validation_batch_index + 1 << "." << endl;

        // ### ### ### --- INITIALIZATION --- ### ### ###

        // initialize arrays which are used below to store the current system states
        vec activations_first_hidden_layer(first_conv_input_channels * M_root * M_root);
        vec activations_second_hidden_layer(first_conv_output_channels * M_root * M_root);
        vec activations_third_hidden_layer(second_conv_output_channels * M_root * M_root);

        vec node_states_first_hidden_layer(first_conv_input_channels * M_root * M_root);
        vec node_states_second_hidden_layer(first_conv_output_channels * M_root * M_root);
        vec node_states_third_hidden_layer(second_conv_output_channels * M_root * M_root);

        double output_activations[P];
        double outputs[P];

        vec g_primes(M);

        // initialize arrays to store deltas and gradient
        vec deltas_first_hidden_layer(first_conv_input_channels * M_root * M_root);
        vec deltas_second_hidden_layer(first_conv_output_channels * M_root * M_root);
        vec deltas_third_hidden_layer(second_conv_output_channels * M_root * M_root);
        double output_deltas[P];

        vec input_weight_gradient(M);
        input_weight_gradient.zeros();
        mat output_weight_gradient(P, second_conv_output_channels * M_root * M_root + 1, fill::zeros);

        field<cube> first_conv_weight_gradient(first_conv_output_channels); // weight_gradient_first_convolution
        for (int i = 0; i < first_conv_output_channels; i++)
        {
            first_conv_weight_gradient(i) = cube(first_conv_input_channels, K, K, fill::zeros);
        }

        field<cube> second_conv_weight_gradient(second_conv_output_channels); // weight_gradient_second_convolution
        for (int i = 0; i < second_conv_output_channels; i++)
        {
            second_conv_weight_gradient(i) = cube(second_conv_input_channels, K, K, fill::zeros);
        }

        // get non-zero diagonal indices for first convolution
        field<cube> temp(first_conv_output_channels);
        for (int i = 0; i < first_conv_output_channels; i++)
        {
            temp(i) = cube(first_conv_input_channels, K, K, fill::ones);
        }
        mat temp1 = unroll_kernel_to_weight_mat(first_conv_output_channels, first_conv_input_channels, temp, M_root);
        vector<int> first_conv_diag_indices;
        first_conv_diag_indices = diagonal_indices(temp1); // Extract non-zero diagonal indices directly from the weight matrix
        std::sort(first_conv_diag_indices.begin(), first_conv_diag_indices.end());

        // get non-zero diagonal indices for second convolution
        temp.set_size(second_conv_output_channels);
        for (int i = 0; i < second_conv_output_channels; i++)
        {
            temp(i) = cube(second_conv_input_channels, K, K, fill::ones);
        }
        temp1 = unroll_kernel_to_weight_mat(second_conv_output_channels, second_conv_input_channels, temp, M_root);
        vector<int> second_conv_diag_indices;
        second_conv_diag_indices = diagonal_indices(temp1); // Extract non-zero diagonal indices directly from the weight matrix
        std::sort(second_conv_diag_indices.begin(), second_conv_diag_indices.end());

        // initialize weights.
        vec input_weights(M);
        input_weights.zeros();
        mat output_weights(P, second_conv_output_channels * M_root * M_root + 1, fill::zeros);

        field<cube> first_conv_hidden_weights(first_conv_output_channels); // for kernels of first convolution
        for (int i = 0; i < first_conv_output_channels; i++)
        {
            first_conv_hidden_weights(i) = cube(first_conv_input_channels, K, K, fill::zeros);
        }

        field<cube> second_conv_hidden_weights(second_conv_output_channels); // for kernels of second convolution
        for (int i = 0; i < second_conv_output_channels; i++)
        {
            second_conv_hidden_weights(i) = cube(second_conv_input_channels, K, K, fill::zeros);
        }

        mat first_hidden_weights_mat(first_conv_output_channels * M_root * M_root, first_conv_input_channels * M_root * M_root, fill::zeros);
        mat second_hidden_weights_mat(second_conv_output_channels * M_root * M_root, second_conv_input_channels * M_root * M_root, fill::zeros);

        initialize_weights(input_weights, first_conv_hidden_weights, second_conv_hidden_weights, first_hidden_weights_mat, second_hidden_weights_mat, output_weights,
                           initial_input_weigt_radius, initial_first_hidden_weigt_radius, initial_second_hidden_weigt_radius, initial_output_weigt_radius, false);

        /*		// weights for test runs.
         mat input_weights_scaled(first_conv_input_channels * M_root * M_root, M + 1, fill::zeros);
         mat output_weights_scaled(P, second_conv_output_channels * M_root * M_root + 1, fill::zeros);

         mat first_hidden_weights_mat_scaled(first_conv_output_channels * M_root * M_root, first_conv_input_channels * M_root * M_root, fill::zeros);
         mat second_hidden_weights_mat_scaled(second_conv_output_channels * M_root * M_root, second_conv_input_channels * M_root * M_root, fill::zeros);



         // dropout mask for weights.
         mat input_weights_mask(first_conv_input_channels * M_root * M_root, M + 1, fill::ones);
         mat output_weights_mask(P, second_conv_output_channels * M_root * M_root + 1, fill::ones);

         mat first_hidden_weights_mat_mask(first_conv_output_channels * M_root * M_root, first_conv_input_channels * M_root * M_root, fill::ones);
         mat second_hidden_weights_mat_mask(second_conv_output_channels * M_root * M_root, second_conv_input_channels * M_root * M_root, fill::ones);

         */

        // ### ### ### --- STOCHASTIC GRADIENT DESCENT TRAINING --- ### ### ###

        // vectors to track training accuracy and validition accuracy (and eventually cosine similarity):
        vector<double> training_accuracy_vector;
        vector<double> accuracy_vector;

        // vector and variables to measure and save cpu time needed for each epoch
        vector<double> time_vector;
        clock_t start;
        clock_t ende;
        double epoch_time;

        // loop over training epochs:
        for (int epoch = 0; epoch < number_of_epochs; ++epoch)
        {

            start = clock();

            // make vector with randomly shuffled indices between 0 and 49999 for each epoch of the stochastic gradient descent
            vector<int> index_vector;
            for (int index = 0; index < (number_of_training_batches - 1) * training_batch_size; ++index)
            {
                index_vector.push_back(index);
            }
            shuffle(begin(index_vector), std::end(index_vector), rng);

            // loop over training steps:
            int step_index = 0;
            for (int index : index_vector)
            {
                ++step_index;
              /*  if (dropout_rate != 0.0)
                {
                	input_weights_mask= mat(first_conv_input_channels * M_root * M_root, M + 1, fill::ones);
                     output_weights_mask= mat(P, second_conv_output_channels * M_root * M_root + 1, fill::ones);
                     first_hidden_weights_mat_mask= mat(first_conv_output_channels * M_root * M_root, first_conv_input_channels * M_root * M_root, fill::ones);
                     second_hidden_weights_mat_mask= mat(second_conv_output_channels * M_root * M_root, second_conv_input_channels * M_root * M_root, fill::ones);

                     for (int m = 0; m < M; ++m){
                     double random_num = uniform(0.0, 1.0);
                     if (random_num < dropout_rate){
                     for (int n = 0; n < first_conv_input_channels * M_root * M_root; ++n){
                     input_weights_mask(n, m) = 0.0;
                     }
                     }
                     }

                     for (int n = 0; n < first_conv_input_channels * M_root * M_root; ++n){
                     double random_num = uniform(0.0, 1.0);
                     if (random_num < dropout_rate){
                     for (int i = 0; i < first_conv_output_channels * M_root * M_root; ++i){
                     first_hidden_weights_mat_mask(i,n)=0.0;
                     }
                     }
                     }

                     for (int n = 0; n < first_conv_output_channels* M_root * M_root; ++n){
                     double random_num = uniform(0.0, 1.0);
                     if (random_num < dropout_rate){
                     for (int j = 0; j < second_conv_output_channels * M_root * M_root; ++j){
                     second_hidden_weights_mat_mask(j,n)=0.0;
                     }
                     }
                     }

                     for (int n = 0; n < second_conv_output_channels * M_root * M_root; ++n){
                     double random_num = uniform(0.0, 1.0);
                     if (random_num < dropout_rate){
                     for (int p = 0; p < P; ++p){
                     output_weights_mask(p,n)=0.0;
                     }
                     }
                     }
                     } */
                    double eta = learning_rate(epoch, step_index, eta_0, eta_1);

                    // select image as input
                    div_t div_result = div(index, training_batch_size);
                    int batch_index = training_batch_indices[div_result.quot];
                    int image_index = div_result.rem;
                    vec input_data = train_images.tube(batch_index, image_index);
                    int label = train_labels[batch_index][image_index];

                    // get target
                    double targets[P];
                    for (int p = 0; p < P; ++p)
                    {
                        targets[p] = input_data(p);
                    }

                    // add noise to input:
                    vec noise = noise_sigma * vec(M, fill::randn);
                    input_data += noise;
                    for (int m = 0; m < M; ++m)
                    {
                        if (input_data[m] < 0.0)
                        {
                            input_data[m] = 0.0;
                        }
                        if (input_data[m] > 1.0)
                        {
                            input_data[m] = 1.0;
                        }
                    }

                    // solve the DDE (or network)
                    clock_t start_solve = clock();
                    if (system_simu == "network")
                    {
                        solve_network(activations_first_hidden_layer, activations_second_hidden_layer, activations_third_hidden_layer, node_states_first_hidden_layer, node_states_second_hidden_layer, node_states_third_hidden_layer, output_activations, outputs, g_primes,
                                      input_data, input_weights, first_hidden_weights_mat, second_hidden_weights_mat, output_weights, theta, alpha,
                                      first_conv_diag_indices, second_conv_diag_indices);
                    }
                    else if

                        (system_simu == "dde_heun")
                    {
                        solve_dde_heun(activations_first_hidden_layer, activations_second_hidden_layer, activations_third_hidden_layer, node_states_first_hidden_layer, node_states_second_hidden_layer, node_states_third_hidden_layer, output_activations, outputs, g_primes,
                                       input_data, input_weights, first_hidden_weights_mat, second_hidden_weights_mat, output_weights, first_conv_diag_indices, second_conv_diag_indices, theta, alpha, N_h);
                    }

                    else
                    {
                        cout << system_simu << " is not a valid value for the system_simu parameter." << endl;
                        abort();
                    }
                    clock_t end_solve = clock();
                    cumulative_solve_time += (end_solve - start_solve) / (double)CLOCKS_PER_SEC;

                    // compute deltas and gradient
                    clock_t start_backprop = clock();
                    vec f_prime_activations_first_hidden_layer = f_prime_matrix(activations_first_hidden_layer);
                    vec f_prime_activations_second_hidden_layer = f_prime_matrix(activations_second_hidden_layer);
                    vec f_prime_activations_third_hidden_layer = f_prime_matrix(activations_third_hidden_layer);

                    if (grad_comp == "backprop_standard")
                    {
                        get_gradient_node_by_node(input_weight_gradient, first_conv_weight_gradient, second_conv_weight_gradient, output_weight_gradient, input_data, node_states_first_hidden_layer, node_states_second_hidden_layer, node_states_third_hidden_layer, f_prime_activations_first_hidden_layer, f_prime_activations_second_hidden_layer, f_prime_activations_third_hidden_layer, g_primes, outputs, targets, first_hidden_weights_mat, second_hidden_weights_mat, output_weights, first_conv_diag_indices, second_conv_diag_indices, theta, alpha);
                    }
                    else
                    {
                        cout << grad_comp << " is not a valid value for the grad_comp option." << endl;
                        abort();
                    }
                    clock_t end_backprop = clock();
                    cumulative_backprop_time += (end_backprop - start_backprop) / (double)CLOCKS_PER_SEC;

                    // perform weight updates

                    input_weights += -eta * input_weight_gradient;

                    // for kernels of first convolution
                    for (int i = 0; i < first_conv_output_channels; i++)
                    {
                        first_conv_hidden_weights(i) += -eta * first_conv_weight_gradient(i);
                    }

                    // for kernels of second convolution
                    for (int i = 0; i < second_conv_output_channels; i++)
                    {
                        second_conv_hidden_weights(i) += -eta * second_conv_weight_gradient(i);
                    }

                    // for hidden weights matrices
                    first_hidden_weights_mat = unroll_kernel_to_weight_mat(first_conv_output_channels, first_conv_input_channels, first_conv_hidden_weights, M_root);
                    second_hidden_weights_mat = unroll_kernel_to_weight_mat(second_conv_output_channels, second_conv_input_channels, second_conv_hidden_weights, M_root);

                    // for output weights
                    output_weights += -eta * output_weight_gradient;
                }

                /*			// weight scaling
                 input_weights_scaled = input_weights / (1.0 - dropout_rate);
                 first_hidden_weights_mat_scaled = first_hidden_weights_mat / (1.0-dropout_rate);
                 second_hidden_weights_mat_scaled = second_hidden_weights_mat / (1.0-dropout_rate);
                 output_weights_scaled = output_weights / (1.0 - dropout_rate);
                 */

                // loop to get accuracy on training set:
                double mse_sum = 0;
                for (int index = 0; index < (number_of_training_batches - 1) * training_batch_size; ++index)
                {
                    // cout << "validation step (on training set)" << index + 1 << endl;

                    // select image as input
                    div_t div_result = div(index, training_batch_size);
                    int batch_index = training_batch_indices[div_result.quot];
                    int image_index = div_result.rem;
                    vec input_data = train_images.tube(batch_index, image_index);
                    int label = train_labels[batch_index][image_index];

                    // get target
                    double targets[P];
                    for (int p = 0; p < P; ++p)
                    {
                        targets[p] = input_data(p);
                    }

                    // add noise to input:
                    vec noise = noise_sigma * vec(M, fill::randn);
                    input_data += noise;
                    for (int m = 0; m < M; ++m)
                    {
                        if (input_data[m] < 0.0)
                        {
                            input_data[m] = 0.0;
                        }
                        if (input_data[m] > 1.0)
                        {
                            input_data[m] = 1.0;
                        }
                    }

                    // solve the DDE (or network)

                    clock_t start_solve = clock();
                    if (system_simu == "network")
                    {
                        solve_network(activations_first_hidden_layer, activations_second_hidden_layer, activations_third_hidden_layer, node_states_first_hidden_layer, node_states_second_hidden_layer, node_states_third_hidden_layer, output_activations, outputs, g_primes,
                                      input_data, input_weights, first_hidden_weights_mat, second_hidden_weights_mat, output_weights, theta, alpha,
                                      first_conv_diag_indices, second_conv_diag_indices);
                    }
                    else if

                        (system_simu == "dde_heun")
                    {
                        solve_dde_heun(activations_first_hidden_layer, activations_second_hidden_layer, activations_third_hidden_layer, node_states_first_hidden_layer, node_states_second_hidden_layer, node_states_third_hidden_layer, output_activations, outputs, g_primes,
                                       input_data, input_weights, first_hidden_weights_mat, second_hidden_weights_mat, output_weights, first_conv_diag_indices, second_conv_diag_indices, theta, alpha,
                                       N_h);
                    }

                    else
                    {
                        cout << system_simu << " is not a valid value for the system_simu parameter." << endl;
                        abort();
                    }
                    clock_t end_solve = clock();
                    cumulative_solve_time += (end_solve - start_solve) / (double)CLOCKS_PER_SEC;

                    double mse = 0;
                    for (int p = 0; p < P; ++p)
                    {
                        mse += pow(outputs[p] - targets[p], 2.0);
                    }
                    mse = mse / (double)P;
                    mse_sum += mse;
                }
                training_accuracy_vector.push_back(mse_sum / 5000.0);

                // loop for validation:
                mse_sum = 0.0;
                for (int index = 0; index < training_batch_size; ++index)
                {
                    // cout << "validation step " << index + 1 << endl;

                    vec input_data = train_images.tube(validation_batch_index, index);
                    int label = train_labels[validation_batch_index][index];

                    // get target
                    double targets[P];
                    for (int p = 0; p < P; ++p)
                    {
                        targets[p] = input_data(p);
                    }

                    // add noise to input:
                    vec noise = noise_sigma * vec(M, fill::randn);
                    input_data += noise;
                    for (int m = 0; m < M; ++m)
                    {
                        if (input_data[m] < 0.0)
                        {
                            input_data[m] = 0.0;
                        }
                        if (input_data[m] > 1.0)
                        {
                            input_data[m] = 1.0;
                        }
                    }

                    // solve the DDE (or network)
                    clock_t start_solve = clock();
                    if (system_simu == "network")
                    {
                        solve_network(activations_first_hidden_layer, activations_second_hidden_layer, activations_third_hidden_layer, node_states_first_hidden_layer, node_states_second_hidden_layer, node_states_third_hidden_layer, output_activations, outputs, g_primes,
                                      input_data, input_weights, first_hidden_weights_mat, second_hidden_weights_mat, output_weights, theta, alpha,
                                      first_conv_diag_indices, second_conv_diag_indices);
                    }
                    else if

                        (system_simu == "dde_heun")
                    {
                        solve_dde_heun(activations_first_hidden_layer, activations_second_hidden_layer, activations_third_hidden_layer, node_states_first_hidden_layer, node_states_second_hidden_layer, node_states_third_hidden_layer, output_activations, outputs, g_primes,
                                       input_data, input_weights, first_hidden_weights_mat, second_hidden_weights_mat, output_weights, first_conv_diag_indices, second_conv_diag_indices, theta, alpha,
                                       N_h);
                    }

                    else
                    {
                        cout << system_simu << " is not a valid value for the system_simu parameter." << endl;
                        abort();
                    }
                    clock_t end_solve = clock();
                    cumulative_solve_time += (end_solve - start_solve) / (double)CLOCKS_PER_SEC;

                    double mse = 0;
                    for (int p = 0; p < P; ++p)
                    {
                        mse += pow(outputs[p] - targets[p], 2.0);
                    }
                    mse = mse / (double)P;
                    mse_sum += mse;
                }
                accuracy_vector.push_back(mse_sum / 1000.0);

                cout << "epoch " << epoch + 1 << ": validation MSE = " << mse_sum / 1000.0 << endl;
                // eventually print weights to file at end of each epoch

                ende = clock();
                epoch_time = ((double)(ende - start)) / (double)CLOCKS_PER_SEC;
                time_vector.push_back(epoch_time);
            }

            // print result to file:
            print_results(results_file_name, validation_batch_index, training_accuracy_vector, accuracy_vector);
        
    }

    // for cpu time measuring
    clock_t end_overall = clock();
    double cpu_time_overall = (end_overall - start_overall) / (double)CLOCKS_PER_SEC;
    double cpu_time_residual = cpu_time_overall - cumulative_solve_time - cumulative_backprop_time;
    double cpu_time_solve_percentage = 100.0 * cumulative_solve_time / cpu_time_overall;
    double cpu_time_backprop_percentage = 100.0 * cumulative_backprop_time / cpu_time_overall;
    double cpu_time_residual_percentage = 100.0 * cpu_time_residual / cpu_time_overall;

    // get current time and date and print to results text file
    auto current_clock = chrono::system_clock::now();
    time_t current_time = chrono::system_clock::to_time_t(current_clock);
    ofstream results_file;
    results_file.open(results_file_name, ios_base::app);
    results_file << endl;
    results_file << endl;
    results_file << "total cpu time (in seconds): " << cpu_time_overall << endl;
    results_file << "cumulative cpu time for solving the DDE or network (in seconds): " << cumulative_solve_time << " (" << cpu_time_solve_percentage << "%)" << endl;
    results_file << "cumulative cpu time for backpropagation (in seconds): " << cumulative_backprop_time << " (" << cpu_time_backprop_percentage << "%)" << endl;
    results_file << "residual cpu time (in seconds): " << cpu_time_residual << " (" << cpu_time_residual_percentage << "%)" << endl;
    results_file << endl;
    results_file << "end of simulation: " << ctime(&current_time);
    results_file.close();

    return 0;
}
