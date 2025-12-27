#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) 
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    float tmp[] = input;
}