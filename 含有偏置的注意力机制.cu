#include <cuda_runtime.h>
#define BLOCK_DIM 32

__global__ void softmax_kernel()
{

}

__global__ void matrix_transpose_kernel(const float* input,float* output,int M,int N)
{
    __shared__ float smem[BLOCK_DIM][BLOCK_DIM];
    
    int block_x = blockIdx.x*BLOCK_DIM;
    int block_y = blockIdx.y*BLOCK_DIM;

    int tid_x = block_x+threadIdx.x;
    int tid_y = block_y+threadIdx.y;

    if(tid_x < N && tid_y < M)
    {
        smem[threadIdx.y][threadIdx.x] = input[tid_y*N+tid_x];
        __syncthreads();
    }

    int x_out = blockIdx.y*BLOCK_DIM+threadIdx.x;
    int y_out = blockIdx.x*BLOCK_DIM+threadIdx.y;

    if(x_out < M && y_out <N)
    {
        output[y_out*M+x_out] = smem[threadIdx.x][threadIdx.y];
    }
}

__global__ void matrix_mult_kernel(const float* input1,const float* input2,int M,int N,int K,float* output)
{
    int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
    int tid_y = blockIdx.y*blockDim.y+threadIdx.y;
    int global_tid = tid_y*N+tid_x;

    __shared__ float smemA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float smemB[BLOCK_DIM][BLOCK_DIM];

    float psum = 0.0f;
    for(int i=0; i<(K+BLOCK_DIM-1)/BLOCK_DIM; ++i)
    {
        if(tid_y < M && i*BLOCK_DIM+threadIdx.x < K)
        {
            smemA[threadIdx.y][threadIdx.x] = input1[tid_y * K+i*BLOCK_DIM+threadIdx.x];
        }
        else
        {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < N && i*BLOCK_DIM+threadIdx.y < K)
        {
            smemB[threadIdx.y][threadIdx.x] = input2[(i*BLOCK_DIM+threadIdx.y)*N+tid_x];
        }
        else
        {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        
        for(int i = 0;i< BLOCK_DIM;++i)
        {
            psum += smemA[i][threadIdx.x]*smemB[threadIdx.y][i];
        }
        __syncthreads();
    }

    if(tid_x < N && tid_y < M)
    {
        output[global_tid] = psum;
    }
}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {}
