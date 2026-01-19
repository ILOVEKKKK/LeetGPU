#include <cuda_runtime.h>

__global__ void nearest_neighbpr_kernel(const float* points,int* indices,int N)
{
    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
    
    if(global_tid < N)
    {
        for(int i = )
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {}
