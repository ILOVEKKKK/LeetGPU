#include <cuda_runtime.h>

__global__ void reduction_kernel(const float* input, float* output, int N)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x*blockDim.x+threadIdx.x;

    //初始化共享内存为现有的input，便于后面归约
    smem[tid] = (global_tid < N)?input[global_tid]:0;
    __syncthreads();

    for(int i = blockDim.x/2;i > 0;i >>= 1)
    {
        if(tid < i)
        {
            smem[tid]+=smem[tid+i];
        }
        __syncthreads();
    }
    //限制只有一个线程能把共享内存中的结果累加到output中
    if(tid == 0)
    {
        atomicAdd(output,smem[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    reduction_kernel<<<gridSize, blockSize, sizeof(float)*blockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}