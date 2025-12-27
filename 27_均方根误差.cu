#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* mse, int N)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;

    if(global_tid < N)
    {
        smem[tid] = (predictions[global_tid]-targets[global_tid])*(predictions[global_tid]-targets[global_tid]);
    }
    else
    {
        smem[tid] = 0.0f;
    }
    __syncthreads();

    for(int i = blockDim.x/2;i>0;i>>=1)
    {
        if(tid < i)
        {
            smem[tid] += smem[tid+i];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        float cur_mse = smem[0]/N;
        atomicAdd(mse,cur_mse);
    }
}
// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) 
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
    size_t shared_mem_size = threadsPerBlock*sizeof(float);
    mse_kernel<<<blocksPerGrid,threadsPerBlock,shared_mem_size>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
