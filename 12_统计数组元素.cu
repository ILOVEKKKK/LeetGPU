#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) 
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx <= N)
    {
        if(input[idx] == K)
        {
            atomicAdd(output,1);//原子操作防止多线程修改output造成错误
        }
    }
}

__global__ void count_equal_kernel_stride_fast(const int* input, int* output, int N, int K) 
{
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    int count = 0;//count是一个线程的私有计数器，负责统计当前线程执行中遇到的满足要求的数组元素数量
    for(int idx = tid;idx < N;idx += stride)//加stride直接跳过了这一轮中其他所有线程执行过的数组元素
    {
        if(input[idx] == K)
        {
            count += 1;
        }
    }
    atomicAdd(output,count);//在一个线程完成操作之后再执行一次原子加法，而不是每遇到一个满足的数组元素就用一次原子加法
}

extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}