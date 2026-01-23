#include <cuda_runtime.h>
#include <float.h>
#define BLOCK_DIM 16

__global__ void QKT_mult_kernel(const float* Q,const float* KT,int N,int d_model,int d_k,int num_head,float* output)
{
    int tid_x = threadIdx.x+blockIdx.x*blockDim.x;
    int tid_y = threadIdx.y+blockIdx.y*blockDim.y;
    int head_id = blockIdx.z;

    const float* Q_head = Q+head_id*d_k;
    const float* KT_head = KT+head_id*d_k*N;
    float* output_head = output+head_id*N;

    __shared__ float smemQ[BLOCK_DIM][BLOCK_DIM];
    __shared__ float smemKT[BLOCK_DIM][BLOCK_DIM];

    float psum = 0.0f;

    for(int i = 0;i<(d_k+BLOCK_DIM-1)/BLOCK_DIM;++i)
    {
        if(tid_y < N && i*BLOCK_DIM+threadIdx.x < d_k)
        {
            smemQ[threadIdx.y][threadIdx.x] = Q_head[tid_y*d_model+i*BLOCK_DIM+threadIdx.x];
        }
        else
        {
            smemQ[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < N && i*BLOCK_DIM+threadIdx.y < d_k)
        {
            smemKT[threadIdx.y][threadIdx.x] = KT_head[(i*BLOCK_DIM+threadIdx.y)*N+tid_x];
        }
        else
        {
            smemKT[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for(int j = 0;j < BLOCK_DIM ;++j)
        {
            psum += smemQ[threadIdx.y][j]*smemKT[j][threadIdx.x];
        }       
    }
    if(tid_x < num_head*N && tid_y < N)
    {
        output_head[tid_y*num_head*N+tid_x] = psum/sqrtf(d_k);
    }
}

__global__ void max_reduction_kernel(const float* input,int N,int d_model,int d_k,int num_head,float* output)
{
    int tid = threadIdx.x;
    int row_id = blockIdx.y;
    int head_id = blockIdx.x;

    const float* input_head = input+head_id*N;
    float* output_head = output+head_id;

    extern __shared__ float smem[];
    float pmax = -FLT_MAX;
    if(row_id < N && head_id < num_head)
    {
        for(int i = tid;i < N;i += blockDim.x)
        {
            pmax = fmaxf(pmax,input_head[row_id*num_head*N+i]);
        }
        smem[tid] = pmax;
        __syncthreads();

        for(int i = blockDim.x/2;i>0;i >>= 1)
        {
            if(tid < i)
            {
                smem[tid] = fmaxf(smem[tid],smem[tid+i]);
            }
            __syncthreads();
        }

        if(tid == 0)
        {
            output_head[row_id*num_head+head_id] = smem[0];
        }
    }
}

__global__ void add_reduction_kernel(const float* input,const float* row_max,int N,int num_head,float* output)
{
    int tid = threadIdx.x;
    int row_id = blockIdx.y;
    int head_id = blockIdx.x;

    const float* input_head = input+head_id*num_head*N;
    float* output_head = output+head_id;

    extern __shared__ float smem[];
    float psum = 0.0f;

    if(row_id < N && head_id < num_head)
    {
        for(int i = tid;i < N;i+=blockDim.x)
        {
            psum += input_head[row_id*num_head*N+i];
        }

        smem[tid] = psum;
        __syncthreads();

        for(int i = blockDim.x/2;i>0;i >>=1)
        {
            if(tid < i)
            {
                smem[tid] += smem[tid+i];
            }
            __syncthreads();
        }

        if(tid == 0)
        {
            output_head[row_id*num_head+head_id] = smem[0];
        }
    }
}

__global__ void softmax_kernel(const float* input,const float* row_sum,const float* row_max,float* output,int N,int num_head)
{
    
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {}
