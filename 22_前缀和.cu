#include <cuda_runtime.h>

//block_scan负责计算一个block内的局部前缀和
__global__ void block_scan(const float* input,float* output,float* block_sums,int N)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
    smem[tid] = (global_tid < N)?input[global_tid]:0.0f;
    __syncthreads();

    //offset是2的倍数，但是每循环offset一次之后，smem中的值就已经包含一部分前缀和了，所以offset取2的倍数是正确的
    for(int offset = 1;offset < blockDim.x;offset <<= 1)
    {
        float temp = 0.0f;
        if(tid >= offset)
        {
            temp = smem[tid]+smem[tid - offset];
        }
        __syncthreads();
        //更新共享内存为offset局部前缀和
        if(tid >= offset)
        {
            smem[tid] = temp;
        }
        __syncthreads();
    }
    //将当前block计算出的局部前缀和写入output
    if(global_tid < N)
    {
        output[global_tid] = smem[tid];
    }
    
    if(tid == blockDim.x-1 && block_sums != nullptr)
    {
        block_sums[blockIdx.x] = smem[tid];
    }
}

__global__ void add_fix(float* block_prefix_sums,int N,float* output)
{
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;

    if(global_tid < N && blockIdx.x>0)
    {
        output[global_tid] += block_prefix_sums[blockIdx.x-1];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) 
{
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);
    float *d_block_sums,*d_block_prefix_sums;
    cudaMalloc(&d_block_sums,gridSize*sizeof(float));
    cudaMalloc(&d_block_prefix_sums,gridSize*sizeof(float));
    block_scan<<<gridSize, blockSize, sharedMemSize>>>(input, output, d_block_sums, N);
    block_scan<<<gridSize, blockSize, gridSize*sizeof(float)>>>(d_block_sums,d_block_prefix_sums,nullptr,gridSize);
    add_fix<<<gridSize, blockSize>>>(d_block_prefix_sums,N,output);
    cudaFree(d_block_sums);
    cudaFree(d_block_prefix_sums);
    cudaDeviceSynchronize();
} 