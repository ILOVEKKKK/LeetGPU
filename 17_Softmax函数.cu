#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) 
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx >= N) return;
    __shared__ float max_val;
    if(threadIdx.x == 0)
    {
        float m = -10000;
        for(int i=0;i<N;i++)
        {
            m = fmaxf(input[i],m);
        }
        max_val = m;
    }
    __syncthreads();

    __shared__ float sum;
    if(threadIdx.x == 0)
    {
        float s = 0;
        for(int i =0;i<N;i++)
        {
            s += expf(input[i]-max_val);
        }
        sum = s;
    }

    __syncthreads();
    if(idx < N)
    {
        output[idx] = expf(input[idx]-max_val)/sum;
    }
}

__global__ void softmax_kernel_fast(const float* input,float* output,int N)
{
    extern __shared__ float shared[];
    float* max_vals = shared;
    float* sums = &shared[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 1. 找最大值
    float local_max = -FLT_MAX;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(input[i], local_max);
    }
    max_vals[tid] = local_max;
    __syncthreads();
    
    // 规约求最大值
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    float max_val = max_vals[0];
    __syncthreads();
    
    // 2. 计算指数和
    float local_sum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += expf(input[i] - max_val);
    }
    sums[tid] = local_sum;
    __syncthreads();
    
    // 规约求总和
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sums[tid] += sums[tid + s];
        }
        __syncthreads();
    }
    float total_sum = sums[0];
    
    // 3. 计算softmax
    if(idx < N) {
        output[idx] = expf(input[idx] - max_val) / total_sum;
    }
}