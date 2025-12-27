#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) 
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < N)
    {
        output[idx] = input[idx]*(1/(1+exp(-input[idx])));
    }
}


