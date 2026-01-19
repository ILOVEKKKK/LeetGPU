#include <cuda_runtime.h>

__global__ void bitonic_sorting_kernel(float* data,int N)
{
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
    int ixj = 
}


extern "C" void solve(float* data,int N)
{

}