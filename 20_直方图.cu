#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input,int* histogram,int N,int num_bins)
{
    //动态声明共享内存
    extern __shared__ int sum[];
    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;

    //初始化局部直方图sum
    for(int i = tid;i < num_bins;i += blockDim.x)
    {
        sum[i] = 0;
    }
    __syncthreads();

    //统计block内的局部直方图sum
    if(global_tid < N)
    {
        if(input[global_tid]>=0 && input[global_tid] < num_bins)
        {
            atomicAdd(&sum[input[global_tid]],1);
        }
    }
    __syncthreads();

    //将每个block的局部直方图sum累加到最终直方图histogram
    for(int i = tid;i < num_bins;i += blockDim.x)
    {
        atomicAdd(&histogram[i],sum[i]);
    }

}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) 
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = num_bins * sizeof(int);
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        input, histogram, N, num_bins
    );
}