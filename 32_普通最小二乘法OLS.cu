#include <cuda_runtime.h>

__global__ void matrix_mult_kernel(const float* A,const float* B,float* C,int M,int N,int K)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y * N+tid_x;

    if(tid_x < N && tid_y < M)
    {
        float sum = 0.0f;
        for(int i = 0;i < K; ++i)
        {
            float a_val = A[tid_y * K+i];
            float b_val = B[i*N+tid_x];
            sum += a_val*b_val;
        }
        C[global_tid] = sum;
    }
}

__global__ void matrix_trans_kernel(const float* A,float* B,int M,int N)
{
    extern __shared__ float smem[];

    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;

    if(tid_x < N && tid_y < M)
    {
        smem[threadIdx.y][threadIdx.x] = A[tid_y*N+tid_x];
    }
    __syncthreads();

    int new_tid_x = blockDim.y*blockIdx.y+threadIdx.y;
    int new_tid_y = blockDim.x*blockIdx.x+threadIdx.x;

    if(new_tid_x < M && new_tid_y < N)
    {
        B[new_tid_y*M+new_tid_x] = smem[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
}

__global__ void matrix_reverse_kernel(const float* A,float* B,float* sum,int M)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;

    smem[tid] = (global_tid < N)?A[global_tid]:0.0f;
    __syncthreads();

    for(int i = blockDim.x/2;i > 0;i >>=1)
    {
        if(tid < i)
        {
            smem[tid]+=smem[tid+i];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        atomicAddd(sum,smem[0])
    }
    __syncthreads();
}


// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) 
{}
