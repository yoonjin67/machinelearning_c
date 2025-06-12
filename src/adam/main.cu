#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>   // For time measurement (e.g., srand seed)
#include <math.h>   // For exp function (used in CPU accuracy calculation)
#include <iostream> // For cout
#include <fstream>  // For file input/output (ifstream)
#include <sstream>  // For string stream (stringstream)
#include <vector>   // For dynamic arrays (std::vector)
#include <string>   // For std::string

// --- Include the header file containing CUDA kernel definitions ---
// This assumes 'logistic_regression_kernels.cuh' is in the same directory or accessible via include paths.
#include "logistic_regression_kernels.cuh"

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
// LOG0_PREVENTION is now defined in logistic_regression_kernels.cuh


// --- Helper function for loading CSV file ---
// This function is used on the host (CPU) side.
void load_iris_csv(const std::string& filename,
                   std::vector<double>& out_samples, // Output vector for sample features
                   std::vector<int>& out_y_true,     // Output vector for true labels
                   int& num_samples,                 // Output: number of samples loaded
                   int& num_features)                // Output: number of features detected
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    num_samples = 0;
    num_features = 0; // Number of features is determined from the first data row

    // Skip the first line (header)
    if (!file.eof()) { // Only try to read if the file is not empty
        std::getline(file, line);
    }

    // Read data rows
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> current_sample_features;
        std::string species_label;

        int current_feature_count = 0;
        // Read each cell separated by commas
        while (std::getline(ss, cell, ',')) {
            if (current_feature_count < 4) { // First 4 columns of Iris dataset are features
                try {
                    current_sample_features.push_back(std::stod(cell)); // Convert string to double
                    current_feature_count++;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid argument for stod: " << cell << " in line: " << line << std::endl;
                    exit(EXIT_FAILURE);
                } catch (const std::out_of_range& e) {
                    std::cerr << "Out of range for stod: " << cell << " in line: " << line << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else { // The last column is the species label
                species_label = cell;
            }
        }

        // Validate if it's a well-formed data line (4 features and a label)
        if (current_feature_count == 4 && !species_label.empty()) {
            if (num_features == 0) { // Determine number of features from the first valid sample
                num_features = current_feature_count;
            }

            // Append features to the main samples vector
            out_samples.insert(out_samples.end(), current_sample_features.begin(), current_sample_features.end());
            // Convert to binary label for logistic regression (Iris-setosa as 1, others as 0)
            if (species_label == "Iris-setosa") {
                out_y_true.push_back(1);
            } else {
                out_y_true.push_back(0);
            }
            num_samples++;
        } else {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
        }
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

// --- Main function (Host code) ---

int main() {
    // 1. Hyperparameter settings
    const std::string CSV_FILENAME = "iris.csv"; // Name of the CSV file
    int NUM_SAMPLES;   // Will be determined dynamically from CSV file
    int NUM_FEATURES;  // Will be determined dynamically from CSV file

    const int NUM_EPOCHS = 5000;   // Number of training epochs (more iterations for smaller datasets)
    const double LEARNING_RATE = 0.005; // Learning rate for Adam optimizer
    const double BETA1 = 0.9;         // Adam hyperparameter beta1
    const double BETA2 = 0.999;       // Adam hyperparameter beta2
    const double EPSILON = 1e-8;      // Adam hyperparameter epsilon (for numerical stability)

    // 2. Allocate CPU memory and load data from Iris CSV file
    std::vector<double> h_samples_vec; // Use std::vector for flexible size
    std::vector<int>    h_y_true_vec;

    load_iris_csv(CSV_FILENAME, h_samples_vec, h_y_true_vec, NUM_SAMPLES, NUM_FEATURES);

    // Get C-style array pointers from std::vectors for CUDA Memcpy operations.
    // These pointers are valid as long as the vectors are in scope and not resized.
    double *h_samples_ptr = h_samples_vec.data();
    int    *h_y_true_ptr = h_y_true_vec.data();

    // Allocate CPU memory for model parameters and Adam moment variables
    double *h_weights = (double*)malloc(NUM_FEATURES * sizeof(double));
    double h_bias = 0.0;

    double *h_first_moment_weights = (double*)calloc(NUM_FEATURES, sizeof(double)); // Initialize to 0
    double *h_second_moment_weights = (double*)calloc(NUM_FEATURES, sizeof(double)); // Initialize to 0
    double h_first_moment_bias = 0.0;
    double h_second_moment_bias = 0.0;

    // Initialize weights and bias with small random values
    srand(time(NULL)); // Seed for random number generation
    for (int i = 0; i < NUM_FEATURES; ++i) {
        h_weights[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05; // Random value between -0.05 and 0.05
    }
    h_bias = ((double)rand() / RAND_MAX) * 0.1 - 0.05;


    // 3. Allocate GPU memory
    double *d_samples, *d_weights, *d_bias_ptr, *d_gradient_weights, *d_gradient_bias_ptr;
    int    *d_y_true;
    double *d_first_moment_weights, *d_second_moment_weights, *d_first_moment_bias_ptr, *d_second_moment_bias_ptr;
    double *d_total_loss; // GPU variable for accumulating total loss

    CUDA_CHECK(cudaMalloc(&d_samples, NUM_SAMPLES * NUM_FEATURES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_true, NUM_SAMPLES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, NUM_FEATURES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bias_ptr, sizeof(double))); // Bias is a single value, but passed as pointer to kernel
    CUDA_CHECK(cudaMalloc(&d_gradient_weights, NUM_FEATURES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gradient_bias_ptr, sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_first_moment_weights, NUM_FEATURES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_second_moment_weights, NUM_FEATURES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_first_moment_bias_ptr, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_second_moment_bias_ptr, sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(double)));


    // 4. Copy data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_samples, h_samples_ptr, NUM_SAMPLES * NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_true, h_y_true_ptr, NUM_SAMPLES * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias_ptr, &h_bias, sizeof(double), cudaMemcpyHostToDevice));

    // Copy initial (zeroed) Adam moment variables from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_first_moment_weights, h_first_moment_weights, NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_second_moment_weights, h_second_moment_weights, NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_first_moment_bias_ptr, &h_first_moment_bias, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_second_moment_bias_ptr, &h_second_moment_bias, sizeof(double), cudaMemcpyHostToDevice));

    // 5. Configure grid and block dimensions for CUDA kernel launches
    // For sample-wise kernels (gradientKernel, binaryCrossentropy)
    int blocks_samples = (NUM_SAMPLES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // For parameter-wise kernel (adam): num_features weights + 1 bias
    int blocks_params = (NUM_FEATURES + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    std::cout << "--- Starting Training ---" << std::endl;

    // 6. Training loop
    for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
        // Zero out gradients and total loss on GPU at the beginning of each epoch
        CUDA_CHECK(cudaMemset(d_gradient_weights, 0, NUM_FEATURES * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_gradient_bias_ptr, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_total_loss, 0, sizeof(double)));

        // 6.1. Launch gradient calculation kernel
        gradientKernel<<<blocks_samples, THREADS_PER_BLOCK>>>(
            d_samples, d_y_true, d_weights, d_bias_ptr,
            NUM_SAMPLES, NUM_FEATURES, d_gradient_weights, d_gradient_bias_ptr
        );
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

        // 6.2. Launch Adam optimizer kernel (parameter update)
        adam<<<blocks_params, THREADS_PER_BLOCK>>>(
            d_weights, d_bias_ptr,
            d_first_moment_weights, d_second_moment_weights,
            d_first_moment_bias_ptr, d_second_moment_bias_ptr,
            d_gradient_weights, d_gradient_bias_ptr,
            NUM_FEATURES, LEARNING_RATE, BETA1, BETA2, EPSILON, epoch // Pass epoch as trained_iteration
        );
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

        // 6.3. Launch loss calculation kernel (periodically or for the first epoch)
        if (epoch % 100 == 0 || epoch == 1) { // Print loss every 100 epochs or on the first epoch
            binaryCrossentropy<<<blocks_samples, THREADS_PER_BLOCK>>>(
                d_samples, d_y_true, d_weights, *d_bias_ptr, // Pass bias value directly
                NUM_SAMPLES, NUM_FEATURES, d_total_loss
            );
            CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

            // Copy total loss value from GPU to CPU and print
            double current_loss = 0.0;
            CUDA_CHECK(cudaMemcpy(&current_loss, d_total_loss, sizeof(double), cudaMemcpyDeviceToHost));
            current_loss /= NUM_SAMPLES; // Calculate average loss
            std::cout << "Epoch " << epoch << ", Loss: " << current_loss << std::endl;
        }
    }

    std::cout << "--- Training Complete ---" << std::endl;

    // 7. Copy final trained weights and bias from GPU to CPU
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, NUM_FEATURES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_bias, d_bias_ptr, sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "\nFinal trained weights: ";
    for (int i = 0; i < NUM_FEATURES; ++i) {
        std::cout << "W[" << i << "] = " << h_weights[i] << " ";
    }
    std::cout << "\nFinal trained bias: B = " << h_bias << std::endl;

    // Predict and calculate accuracy on the training dataset (on CPU)
    // Define a CPU-side sigmoid function as __device__ functions cannot be called from host.
    auto cpu_sigmoid = [](double z) { return 1.0 / (1.0 + exp(-z)); };

    int correct_predictions = 0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        double z_cpu = h_bias;
        for (int j = 0; j < NUM_FEATURES; ++j) {
            z_cpu += h_weights[j] * h_samples_vec[i * NUM_FEATURES + j];
        }
        double y_pred_cpu = cpu_sigmoid(z_cpu); // Calculate sigmoid on CPU
        int predicted_class = (y_pred_cpu > 0.5) ? 1 : 0;
        if (predicted_class == h_y_true_vec[i]) { // Compare with true label from h_y_true_vec
            correct_predictions++;
        }
    }
    double accuracy = (double)correct_predictions / NUM_SAMPLES * 100.0;
    std::cout << "Training dataset accuracy: " << accuracy << "% (" << correct_predictions << "/" << NUM_SAMPLES << ")" << std::endl;


    // 8. Free GPU memory
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_y_true));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias_ptr));
    CUDA_CHECK(cudaFree(d_gradient_weights));
    CUDA_CHECK(cudaFree(d_gradient_bias_ptr));
    CUDA_CHECK(cudaFree(d_first_moment_weights));
    CUDA_CHECK(cudaFree(d_second_moment_weights));
    CUDA_CHECK(cudaFree(d_first_moment_bias_ptr));
    CUDA_CHECK(cudaFree(d_second_moment_bias_ptr));
    CUDA_CHECK(cudaFree(d_total_loss));

    // 9. Free CPU memory
    // std::vectors (h_samples_vec, h_y_true_vec) are automatically deallocated when they go out of scope.
    // Only free memory allocated with malloc/calloc.
    free(h_weights);
    free(h_first_moment_weights);
    free(h_second_moment_weights);

    return 0;
}
