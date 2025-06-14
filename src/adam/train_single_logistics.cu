#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>   
#include <math.h>   
#include <iostream> 
#include <fstream>  
#include <sstream>  
#include <vector>   
#include <string>   
#include <algorithm>
#define FEATURES_SIZE 4
#include "adam.cuh"

// --- Macro Definitions ---
// CUDA error checking macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), \
                    __FILE__, __LINE__);                                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define THREADS_PER_BLOCK 512

void load_iris_csv(const std::string& filename,
                   std::vector<double>& out_samples,
                   std::vector<double>& out_y_true, 
                   int& num_samples,                
                   int& num_features)               
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    num_samples = 0;
    num_features = FEATURES_SIZE; 

    std::string cell;
    std::getline(file, line); // Skip header line

    while (std::getline(file, line)) {
        std::stringstream ss(line);  // <== Move this inside the loop
        std::vector<double> current_sample_features;
        std::string cell;
        std::string species_label;
    
        // Skip the ID column, then read 4 features
        std::getline(ss, cell, ','); // Skip ID column
        for (int i = 0; i < FEATURES_SIZE; ++i) {
            if (!std::getline(ss, cell, ',')) {
                std::cerr << "Warning: Skipping malformed line (missing feature column at index " << i << "): " << line << std::endl;
                current_sample_features.clear();
                break;
            }
            try {
                current_sample_features.push_back(std::stod(cell));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument for std: " << cell << " in line: " << line << std::endl;
                current_sample_features.clear();
                break;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range for std: " << cell << " in line: " << line << std::endl;
                current_sample_features.clear();
                break;
            }
        }
    
        if (current_sample_features.size() != FEATURES_SIZE) {
            continue;
        }
    
        // Read the species label
        if (!std::getline(ss, cell)) {
            std::cerr << "Warning: Skipping malformed line (missing species label or extra data): " << line << std::endl;
            continue;
        }
        species_label = cell;
    
        out_samples.insert(out_samples.end(), current_sample_features.begin(), current_sample_features.end());
    
        if (species_label == "Iris-setosa") {
            out_y_true.push_back(1.0);
        } else {
            out_y_true.push_back(0.0);
        }
        num_samples++;
    }

    
    file.close();
    if (num_samples == 0) { // Error if no valid samples were loaded
        std::cerr << "Error: No valid data samples found in CSV file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "CSV file loaded successfully: " << filename << std::endl;
    std::cout << "Total samples: " << num_samples << std::endl;
    std::cout << "Number of features: " << num_features << std::endl;
}

// --- Function to save trained weights and bias to a file ---
void save_weights(
    const std::string& filename,
    const double* weights,
    double bias,
    int num_features
) 
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to save weights: " << filename << std::endl;
        return;
    }
    // Save number of features, then weights, then bias
    file << num_features << std::endl;
    for (int i = 0; i < num_features; ++i) {
        file << weights[i] << (i == num_features - 1 ? "" : " "); // Space separated, no trailing space
    }
    file << std::endl;
    file << bias << std::endl;
    file.close();
    std::cout << "\nWeights and bias saved to: " << filename << std::endl;
}

// Returns true on success, false on failure (file not found, malformed, or feature mismatch)
bool load_weights(const std::string& filename, double* weights_out, double& bias_out, int num_features_expected) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file to load weights: " << filename << std::endl;
        return false; // Indicate failure (e.g., file not found)
    }

    int loaded_num_features;
    file >> loaded_num_features;
    if (file.fail() || loaded_num_features <= 0) {
        std::cerr << "Error: Malformed weights file (could not read features) in: " << filename << std::endl;
        file.close();
        return false;
    }

    if (loaded_num_features != num_features_expected) {
        std::cerr << "Error: Mismatch in number of features when loading weights from " << filename
                  << ". Loaded: " << loaded_num_features << ", Expected from data: " << num_features_expected << std::endl;
        file.close();
        return false;
    }

    for (int i = 0; i < loaded_num_features; ++i) {
        file >> weights_out[i];
        if (file.fail()) {
            std::cerr << "Error: Malformed weights file (could not read weight " << i << ") in: " << filename << std::endl;
            file.close();
            return false;
        }
    }
    file >> bias_out;
    if (file.fail()) {
        std::cerr << "Error: Malformed weights file (could not read bias) in: " << filename << std::endl;
        file.close();
        return false;
    }

    file.close();
    std::cout << "Weights and bias loaded from: " << filename << std::endl;
    return true; // Indicate success
}


// --- Main function (Host code) ---
int main(int argc, char* argv[]) {
    // Hyperparameter settings
    std::string TRAIN_CSV_FILENAME = "train.csv"; 
    std::string TEST_CSV_FILENAME = "test.csv";  
    std::string WEIGHTS_SAVE_FILENAME = "model_weights.txt"; 

    int NUM_TRAIN_SAMPLES;
    int NUM_TRAIN_FEATURES;

    int NUM_TEST_SAMPLES;
    int NUM_TEST_FEATURES;

    const int NUM_EPOCHS = 10;
    const double LEARNING_RATE = 0.001;
    const double BETA1 = 0.9;
    const double BETA2 = 0.999;
    const double EPSILON = 1e-8;

    // Determine operation mode based on command line arguments
    bool perform_training = true;
    std::string eval_weights_path = "";
    std::string training_warm_start_weights_path = "";

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--evaluate" && i + 1 < argc) {
            perform_training = false; 
            eval_weights_path = argv[++i]; 
        } else if (std::string(argv[i]) == "--load_weights" && i + 1 < argc) {
            // This argument is specifically for loading weights at the start of training (warm-start)
            training_warm_start_weights_path = argv[++i];
        } else if (std::string(argv[i]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [--evaluate <weights_file>] [--load_weights <weights_file>]" << std::endl;
            std::cout << "  --evaluate <weights_file> : Load weights and evaluate on test data without training." << std::endl;
            std::cout << "  --load_weights <weights_file> : Load weights and continue training from them (warm-start)." << std::endl;
            std::cout << "  --train <train_file> : Load train data and continue training from them (warm-start)." << std::endl;
            std::cout << "  --test <test_file> : Load test data and continue testing from them (warm-start)." << std::endl;
            std::cout << "  --save <save_file> : Save weights on specified path." << std::endl;
            return 0; // Exit after showing help
        } else if(std::string(argv[i]) == "--train" && i + 1 < argc) {
            TRAIN_CSV_FILENAME = std::string(argv[++i]);
        } else if(std::string(argv[i]) == "--test" && i + 1 < argc) {
            TEST_CSV_FILENAME = std::string(argv[++i]);
        } else  if(std::string(argv[i]) == "--save" && i + 1 < argc) {
            WEIGHTS_SAVE_FILENAME = std::string(argv[++i]);
        }
    }

    // Allocate CPU memory and load training data from CSV file
    std::cout << "--- Loading Training Data ---" << std::endl;
    std::vector<double> h_train_samples_vec;
    std::vector<double> h_train_y_true_vec;
    load_iris_csv
    (
            TRAIN_CSV_FILENAME,
            h_train_samples_vec,
            h_train_y_true_vec,
            NUM_TRAIN_SAMPLES, 
            NUM_TRAIN_FEATURES
    );

    // Load test data from CSV file
    std::cout << "\n--- Loading Test Data ---" << std::endl;
    std::vector<double> h_test_samples_vec;
    std::vector<double> h_test_y_true_vec;
    load_iris_csv
    (
     TEST_CSV_FILENAME,
     h_test_samples_vec,
     h_test_y_true_vec,
     NUM_TEST_SAMPLES,
     NUM_TEST_FEATURES
     );

    // Basic check to ensure feature counts match
    if (NUM_TRAIN_FEATURES != NUM_TEST_FEATURES) {
        std::cerr << "Error: Number of features in training data (" << NUM_TRAIN_FEATURES
                  << ") does not match test data (" << NUM_TEST_FEATURES << ")." << std::endl;
        exit(EXIT_FAILURE);
    }
    int NUM_FEATURES = NUM_TRAIN_FEATURES; // Authoritative number of features for the model

    // Allocate CPU memory for model parameters
    double *h_weights = (double*)malloc(NUM_FEATURES * sizeof(double));
    double h_bias = 0.0;
    bool initial_weights_loaded_successfully = false;

    // Load or initialize weights based on mode
    if (!eval_weights_path.empty()) { 
        std::cout << "\n--- Running in Evaluation Mode ---" << std::endl;
        perform_training = false; 
        initial_weights_loaded_successfully = load_weights(eval_weights_path, h_weights, h_bias, NUM_FEATURES);
        if (!initial_weights_loaded_successfully) {
            std::cerr << "Error: Failed to load weights for evaluation from " << eval_weights_path << std::endl;
            exit(EXIT_FAILURE); // Exit if weights can't be loaded for evaluation
        }
    } else if (!training_warm_start_weights_path.empty()) { // --- Training with Warm Start ---
        std::cout << "\n--- Starting Training with Warm Start ---" << std::endl;
        initial_weights_loaded_successfully = load_weights(training_warm_start_weights_path, h_weights, h_bias, NUM_FEATURES);
        if (!initial_weights_loaded_successfully) {
            std::cerr << "Error: Failed to load warm-start weights from " << training_warm_start_weights_path << ". Exiting." << std::endl;
            exit(EXIT_FAILURE); // Warm-start failed, can't continue training as intended
        }
    } else { 
        std::cout << "\nInitializing weights and bias with random values..." << std::endl;
        srand(time(NULL));
        for (int i = 0; i < NUM_FEATURES; ++i) {
            h_weights[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        }
        h_bias = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    }

    // Allocate Adam moment variables only if performing training
    double *h_first_moment_weights = nullptr;
    double *h_second_moment_weights = nullptr;
    double h_first_moment_bias = 0.0;
    double h_second_moment_bias = 0.0;

    if (perform_training) {
        h_first_moment_weights = (double*)calloc(NUM_FEATURES, sizeof(double)); // Initialize to 0
        h_second_moment_weights = (double*)calloc(NUM_FEATURES, sizeof(double)); // Initialize to 0
        // h_first_moment_bias and h_second_moment_bias are already 0.0
    }

    // Allocate GPU memory for model parameters (always needed)
    double *d_weights, *d_bias_ptr;
    CUDA_CHECK(cudaMalloc(&d_weights, NUM_FEATURES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bias_ptr, sizeof(double)));

    // Copy initial weights and bias to GPU
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias_ptr, &h_bias, sizeof(double), cudaMemcpyHostToDevice));


    if (perform_training) {
        // GPU memory allocation and data transfer for TRAINING
        double *d_train_samples;
        int *d_train_y_true;
        double *d_gradient_weights, *d_gradient_bias_ptr;
        double *d_first_moment_weights_gpu, *d_second_moment_weights_gpu;
        double *d_first_moment_bias_ptr_gpu, *d_second_moment_bias_ptr_gpu;
        double *d_total_loss;

        CUDA_CHECK(cudaMalloc(
                    &d_train_samples,
                    NUM_TRAIN_SAMPLES * NUM_FEATURES * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
                    &d_train_y_true,
                    NUM_TRAIN_SAMPLES * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
                    &d_gradient_weights, 
                    NUM_FEATURES * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
                    &d_gradient_bias_ptr, sizeof(double)));
        CUDA_CHECK(cudaMalloc(
                    &d_first_moment_weights_gpu, 
                    NUM_FEATURES * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_second_moment_weights_gpu, NUM_FEATURES * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_first_moment_bias_ptr_gpu, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_second_moment_bias_ptr_gpu, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(double)));

        // Copy training data and initial Adam moment variables to GPU
        CUDA_CHECK(cudaMemcpy(d_train_samples, h_train_samples_vec.data(), 
                    NUM_TRAIN_SAMPLES * NUM_FEATURES * sizeof(double), 
                    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_train_y_true, h_train_y_true_vec.data(),
                    NUM_TRAIN_SAMPLES * sizeof(double),
                    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_first_moment_weights_gpu, h_first_moment_weights,
                    NUM_FEATURES * sizeof(double),
                    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_second_moment_weights_gpu, h_second_moment_weights,
                    NUM_FEATURES * sizeof(double), 
                    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_first_moment_bias_ptr_gpu, &h_first_moment_bias,
                    sizeof(double),
                    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_second_moment_bias_ptr_gpu, &h_second_moment_bias,
                    sizeof(double),
                    cudaMemcpyHostToDevice));


        // Configure grid and block dimensions for CUDA kernel launches (for training)
        int blocks_samples = (NUM_TRAIN_SAMPLES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int blocks_params = (NUM_FEATURES + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        std::cout << "\n--- Starting Training ---" << std::endl;

        // 6. Training loop
        for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
            // Zero out gradients and total loss on GPU at the beginning of each epoch
            CUDA_CHECK(cudaMemset(d_gradient_weights, 0, NUM_FEATURES * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_gradient_bias_ptr, 0, sizeof(double)));
            CUDA_CHECK(cudaMemset(d_total_loss, 0, sizeof(double)));

            // 6.1. Launch gradient calculation kernel
            gradient_kernel<<<blocks_samples, THREADS_PER_BLOCK>>>(
                d_train_samples, d_train_y_true, d_weights, d_bias_ptr,
                NUM_TRAIN_SAMPLES, NUM_FEATURES, d_gradient_weights, d_gradient_bias_ptr
            );
            CUDA_CHECK(cudaGetLastError());

            // 6.2. Launch Adam optimizer kernel (parameter update)
            adam<<<blocks_params, THREADS_PER_BLOCK>>>(
                d_weights, d_bias_ptr,
                d_first_moment_weights_gpu, d_second_moment_weights_gpu,
                d_first_moment_bias_ptr_gpu, d_second_moment_bias_ptr_gpu,
                d_gradient_weights, d_gradient_bias_ptr,
                NUM_FEATURES, LEARNING_RATE, BETA1, BETA2, EPSILON, epoch // Pass epoch as trained_iteration
            );
            CUDA_CHECK(cudaGetLastError());

            // 6.3. Launch loss calculation kernel (periodically or for the first epoch)
            if (epoch % 2 == 0 || epoch == 1) { // Print loss every 5 epochs or on the first epoch
                double current_bias_value_on_host = 0.0;
                CUDA_CHECK(cudaMemcpy(&current_bias_value_on_host, d_bias_ptr, sizeof(double), cudaMemcpyDeviceToHost));

                binary_crossentropy<<<blocks_samples, THREADS_PER_BLOCK>>>(
                    d_train_samples, d_train_y_true, d_weights, current_bias_value_on_host,
                    NUM_TRAIN_SAMPLES, NUM_FEATURES, d_total_loss
                );
                CUDA_CHECK(cudaGetLastError());

                double current_loss = 0.0;
                CUDA_CHECK(cudaMemcpy(&current_loss, d_total_loss, sizeof(double), cudaMemcpyDeviceToHost));
                current_loss /= NUM_TRAIN_SAMPLES; // Calculate average loss
                std::cout << "Epoch " << epoch << ", Loss: " << current_loss << std::endl;
            }
        }

        std::cout << "--- Training Complete ---" << std::endl;

        // Copy final trained weights and bias from GPU to CPU
        CUDA_CHECK(cudaMemcpy(h_weights, d_weights, NUM_FEATURES * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_bias, d_bias_ptr, sizeof(double), cudaMemcpyDeviceToHost));

        std::cout << "\nFinal trained weights: ";
        for (int i = 0; i < NUM_FEATURES; ++i) {
            std::cout << "W[" << i << "] = " << h_weights[i] << " ";
        }
        std::cout << "\nFinal trained bias: B = " << h_bias << std::endl;

        // --- Save trained weights and bias to a file ---
        save_weights(WEIGHTS_SAVE_FILENAME, h_weights, h_bias, NUM_FEATURES);

        // Free training specific GPU memory
        CUDA_CHECK(cudaFree(d_train_samples));
        CUDA_CHECK(cudaFree(d_train_y_true));
        CUDA_CHECK(cudaFree(d_gradient_weights));
        CUDA_CHECK(cudaFree(d_gradient_bias_ptr));
        CUDA_CHECK(cudaFree(d_first_moment_weights_gpu));
        CUDA_CHECK(cudaFree(d_second_moment_weights_gpu));
        CUDA_CHECK(cudaFree(d_first_moment_bias_ptr_gpu));
        CUDA_CHECK(cudaFree(d_second_moment_bias_ptr_gpu));
        CUDA_CHECK(cudaFree(d_total_loss));

    } else { // Evaluation mode - h_weights and h_bias already loaded
        std::cout << "\n--- Skipping Training (Evaluation Mode) ---" << std::endl;
    }

    // --- Evaluate model on the TEST dataset (on CPU) ---
    // This part runs in both training and evaluation modes.
    std::cout << "\n--- Evaluating Model on Test Data ---" << std::endl;

    // Define a CPU-side sigmoid function as __device__ functions cannot be called from host.
    auto cpu_sigmoid = [](double z) { return 1.0 / (1.0 + exp(-z)); };

    int correct_predictions_test = 0;
    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        double z_cpu = h_bias;
        for (int j = 0; j < NUM_TEST_FEATURES; ++j) {
            z_cpu += h_weights[j] * h_test_samples_vec[i * NUM_TEST_FEATURES + j]; // Use test samples
        }
        double y_pred_cpu = cpu_sigmoid(z_cpu); // Calculate sigmoid on CPU
        int predicted_class = (y_pred_cpu > 0.5) ? 1 : 0;
        if (predicted_class == h_test_y_true_vec[i]) { // Compare with true label from test data
            correct_predictions_test++;
        }
    }
    double accuracy_test = (double)correct_predictions_test / NUM_TEST_SAMPLES * 100.0;
    std::cout << "Test dataset accuracy: " << accuracy_test << "% (" << correct_predictions_test << "/" << NUM_TEST_SAMPLES << ")" << std::endl;


    // 8. Free general GPU memory (d_weights, d_bias_ptr are always allocated)
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias_ptr));

    // 9. Free CPU memory
    free(h_weights); // Always free h_weights as it's always malloc'd
    if (perform_training) { // Only free if allocated
        free(h_first_moment_weights);
        free(h_second_moment_weights);
    }

    return 0;
}
