#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A,const float* B,float* result,int N)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
    //共享内存预加载向量每个元素的乘积
    smem[tid] = (global_tid < N)?A[global_tid]*B[global_tid]:0.0f;
    __syncthreads();
    //归约求和得到点积结果
    for(int i = blockDim.x/2;i>0;i>>=1)
    {
        if(tid < i)
        {
            smem[tid]+=smem[tid+i];
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        atomicAdd(result,smem[0]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) 
{
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dot_kernel<<<gridSize, blockSize, sizeof(float)*blockSize>>>(A,B,result,N);
    cudaDeviceSynchronize();
}