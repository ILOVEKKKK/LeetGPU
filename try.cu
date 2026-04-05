#include <cuda_runtime.h>
#define BLOCKDIM 32

__global__ void mean_kernel(const float* input,float* output,int N,int C)
{
    int tid_y = threadIdx.y;
    int col_id = blockIdx.x;

    extern __shared__ float smem[];
    float psum = 0.0f;

    for(int i =tid_y;i < N;i+=blockDim.y)
    {
        psum += input[i*C+col_id];
    } 
    smem[tid_y] = psum;
    __syncthreads();

    for(int i = blockDim.y/2;i>0;i>>=1)
    {
        if(tid_y < i)
        {
            smem[tid_y] += smem[tid_y+i];
        }
        __syncthreads();
    }

    if(tid_y == 0)
    {
        output[col_id] = smem[0]/N;
    }
}

__global__ void mean_shared_kernel(const float* input, float* output, int N, int C)
{
    int tid_y = threadIdx.y;
    int tid_x = threadIdx.x;
    int col_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float smem[BLOCKDIM][BLOCKDIM];
    
    // 1. 默认初始化 psum 为 0
    // 对于越界线程，psum 保持为 0，这很重要！
    float psum = 0.0f;

    // 2. 只有合法的线程才去读 Global Memory (Masking)
    if(col_id < C) 
    {
        for(int i = tid_y; i < N; i += blockDim.y)
        {
            psum += input[i * C + col_id]; 
        }
    }

    // 3. 【关键】无论是否越界，都写入 smem
    // 越界线程写入 0.0f，不会破坏归约结果
    smem[tid_y][tid_x] = psum;

    // 4. 【关键】同步屏障必须在 if 外面！
    // 这样 0-31 号线程都能走到这里，不会死锁
    __syncthreads();

    // 5. 归约 (所有线程都参与，或者依然保持同步)
    for(int i = blockDim.y / 2; i > 0; i >>= 1)
    {
        if(tid_y < i)
        {
            // 越界线程在这里加的是 0 + 0 = 0，安全
            smem[tid_y][tid_x] += smem[tid_y + i][tid_x];
        }
        __syncthreads(); // 这个同步也是安全的，因为所有线程都能走到
    }

    // 6. 写回结果时，再做一次边界检查
    if(tid_y == 0 && col_id < C)
    {
        output[col_id] = smem[0][tid_x] / (float)N;
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps)
{}
