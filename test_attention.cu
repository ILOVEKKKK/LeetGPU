#include <cuda_runtime.h>
#include <float.h>
#define BLOCKDIM 32

__global__ void mat_mul_kernel(const float* Q,const float* K,float* output,int m,int k,int n)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;

    __shared__ float smemQ[BLOCKDIM][BLOCKDIM];
    __shared__ float smemK[BLOCKDIM][BLOCKDIM];

    float psum = 0.0f;

    for(int i = 0;i < (k+BLOCKDIM-1)/BLOCKDIM;++i)
    {
        if(tid_y < m && i*BLOCKDIM+threadIdx.x < k)
        {
            smemQ[threadIdx.y][threadIdx.x] = Q[tid_y*k+i*BLOCKDIM+threadIdx.x];
        }
        else
        {
            smemQ[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < n && i*BLOCKDIM+threadIdx.y < k)
        {
            smemK[threadIdx.y][threadIdx.x] = K[(i*BLOCKDIM+threadIdx.y)*n+tid_x];
        }
        else
        {
            smemK[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
    }

    for(int i = 0;i<BLOCKDIM;++i)
    {
        psum += smemQ[threadIdx.y][i]*smemK[i][threadIdx.x];
    }

    if(tid_x < n && tid_y < m)
    {
        output[tid_y*n+tid_x] = psum;
    }
}

__global__ void mat_transpose_kernel(const float* input,float* output,int m,int n)
{
    int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
    int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

    __shared__ float smem[BLOCKDIM][BLOCKDIM];

    if(tid_x < n && tid_y < m)
    {
        smem[threadIdx.y][threadIdx.x] = input[tid_y*n+tid_x];
        __syncthreads();
    }

    int x_out = blockDim.y*blockIdx.y+threadIdx.x;
    int y_out = blockDim.x*blockIdx.x+threadIdx.y;

    if(x_out < m && y_out < n)
    {
        output[y_out * m+x_out] = smem[threadIdx.x][threadIdx.y];
    }
}

__global__ void row_max_kernel(const float* input,float* row_max,int m,int n)
{
    int tid_x = threadIdx.x;
    int row_id = blockIdx.x;
    extern __shared__ float smem[];

    if(row_id >= m) return;
    float pmax = -FLT_MAX;
    for(int i = tid_x;i < n;i += blockDim.x)
    {
        pmax = fmaxf(smem[i],input[row_id*n+i]);
    }
    smem[tid_x] = pmax;
    __syncthreads();

    for(int i = blockDim.x/2;i>0;i>>=1)
    {
        if(tid_x < i)
        {
            smem[tid_x] = fmaxf(smem[tid_x],smem[tid_x+i]);
        }
        __syncthreads();
    }

    if(tid_x == 0)
    {
        row_max[row_id] = smem[0];
    }
}

__global__ void row_sum_kernel(const float* input,const float* row_max,float* row_sum,int m,int n)
{
    int tid_x = threadIdx.x;
    int row_id = blockIdx.x;

    if(row_id >= m)return;
    extern __shared__ float smem[];

    float psum = 0.0f;
    for(int i = tid_x;i < n;i += blockDim.x)
    {
        
    }


}