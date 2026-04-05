#include <cuda_runtime.h>

__global__ void montecarlo_kernel(const float* y_samples, float* result, float a, float b, int n_samples)
{
    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
    
    extern __shared__ float smem[];
    smem[tid] = global_tid < n_samples?y_samples[global_tid]:0.0f;
    __syncthreads();

    for(int i = blockDim.x/2;i > 0;i >>= 1)
    {
        if(tid < i)
        {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        atomicAdd(result,(b-a)*(1.0f/n_samples)*smem[tid]);
    }
} 

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples)
{
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n_samples+threadsPerBlock.x-1)/threadsPerBlock.x);
    size_t shared_mem_size = threadsPerBlock.x*sizeof(float);
    montecarlo_kernel<<<blocksPerGrid,threadsPerBlock,shared_mem_size>>>(y_samples,result,a,b,n_samples);
    cudaDeviceSynchronize();
}