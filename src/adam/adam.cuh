#include <cuda_runtime.h>
#include <math.h> 

#define LOG0_PREVENTION 1e-9

__device__ double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

__device__ double proba(
    double *x,
    double *weights,
    double bias,
    int num_features
)
{
    double z_gpu = bias;
    for(int i = 0; i < num_features; ++i) {
        z_gpu += weights[i] * x[i];
    }
    return sigmoid(z_gpu);
}

__global__ void gradientKernel(
    double *samples,
    int    *y_true,
    double *weights,
    double *bias,
    int num_samples,
    int num_features,
    double *gradient_weights,
    double *gradient_bias
); 

__global__ void binaryCrossentropy(
    double *samples,
    int    *y_true,
    double *weights,
    double bias,
    int num_samples,
    int num_features,
    double *total_loss
); 

__global__ void adam
(
    double *weights,
    double *bias,
    double *first_moment_weights,
    double *second_moment_weights,
    double *first_moment_bias,
    double *second_moment_bias,
    double *gradient_weights,
    double *gradient_bias,
    int    num_features,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    int    trained_iteration
); 
