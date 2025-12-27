#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) 
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < halfN)
    {
        output[idx] = input[idx]*(1/(1+exp(-input[idx])))*input[idx+halfN];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}