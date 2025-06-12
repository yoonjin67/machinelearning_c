#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define THREADS_PER_BLOCK 512
#define SIZE (2048 * 2048)
#define NUM_TASKS 4
#define LOG0_PREVENTION 1e-9
//<<<SIZE / THREADS_PER_BLOCK * NUM_TASKS, THREADS_PER_BLOCK>>>



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
    int   *y_true,
    double *weights, 
    double *bias,
    int num_samples, //this indicates how many samples are loaded
    int num_features, //this indicates dimension of X
    double *gradient_weights, //1st dimension array for storing gradient weights
    double *gradient_bias     //pointer for gradient bias (which is not 1st dimension array)
) 
{
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(sampleIdx >= num_samples) return; //to prevent out of range
    const double* current_sample = &samples[sampleIdx * num_features];
    double y_pred = proba(current_sample, weights, *bias, num_features);
    double error = y_pred - y_true[sampleIdx];
    for(int nth_feature = 0; nth_feature < num_features; nth_feature++) {
    //need to use atomicAdd to retain atomicity; when using thread this must be done.
        atomicAdd(&gradient_weights[nth_feature] ,error * current_sample[nth_feature]);
    }
    atomicAdd(gradient_bias, error);
}


__global__ void binaryCrossentropy(
    double *samples,
    int    *y_true,
    double *weights,
    double bias,
    int num_samples,
    int num_features,
    double *total_loss
)
{
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(sampleIdx >= num_samples) return; //to prevent out of range
    double y_pred = proba(&samples[sampleIdx * num_features], weights, bias, num_features);
    y_pred = fmax(fmin(y_pred, 1.0 - LOG0_PREVENTION), LOG0_PREVENTION);
    atomicAdd(total_loss, -((double)y_true[sampleIdx] * log(y_pred) + (1.0 - (double)y_true[sampleIdx]) * log(1.0 - y_pred)));
}

__global__ void adam
(
    double *weights,
    double *bias,
    double *first_moment_weights,   //moment of adam. used for input and output
    double *second_moment_weights,  //moment 2
    double *first_moment_bias,
    double *second_moment_bias,
    double *gradient_weights,
    double *gradient_bias,
    int    num_features,
    double learning_rate,
    double beta1,
    double beta2, //hyperparameters
    double epsilon, //stabilizes adam
    int    trained_iteration

) 
{
    int parameterIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(parameterIdx < num_features) { //of course, don't access more than received.
        first_moment_weights[parameterIdx] = beta1 * first_moment_weights[parameterIdx] + (1.0 - beta1) * gradient_weights[parameterIdx];
        second_moment_weights[parameterIdx] = beta2 * second_moment_weights[parameterIdx] + (1.0 - beta2) * (gradient_weights[parameterIdx] * gradient_weights[parameterIdx]);
                //second moment adds square of beta2. #1
        double first_moment_adjusted  = first_moment_weights[parameterIdx] / (1.0 - pow(beta1, (double)trained_iteration));
        double second_moment_adjusted = second_moment_weights[parameterIdx] / (1.0 - pow(beta2, (double)trained_iteration));
        // Update weights via adam formula
        weights[parameterIdx] -= learning_rate * first_moment_adjusted / (sqrt(second_moment_adjusted) + epsilon);
        //gets square root of calculated moment. If not, first and second would be too far. it's quite abstract understanding. see paper
        } else if(parameterIdx == num_features) { //in here, we can finalize weight calculation. as it is last parameter.
            *first_moment_bias = beta1 * (*first_moment_bias) + (1.0 - beta1) * (*gradient_bias);
            *second_moment_bias = beta2 * (*second_moment_bias) + (1.0 - beta2) * ((*gradient_bias) * (*gradient_bias));
            double adjusted_first_moment_bias = (*first_moment_bias) / (1.0 - pow(beta1, (double)trained_iteration));
            double adjusted_second_moment_bias = (*second_moment_bias) / (1.0 - pow(beta2, (double)trained_iteration));
            *bias -= learning_rate * adjusted_first_moment_bias / (sqrt(adjusted_second_moment_bias) + epsilon);
    }
}
