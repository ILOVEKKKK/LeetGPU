#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define BLOCK_DIM 16

__global__ void matrix_mult_fp16_kernel(const half* A,const half*B,half* C,int BATCH,int M,int N,int K)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int tid_z = blockIdx.z;
    int global_tid = tid_z * M * N+tid_y * N+tid_x;

    float psum = 0.0f;

    __shared__ float smemA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float smemB[BLOCK_DIM][BLOCK_DIM];

    for(int i = 0;i < (K+BLOCK_DIM-1)/BLOCK_DIM;++i)
    {
        if(tid_y < M && i*BLOCK_DIM+threadIdx.x < K && tid_z < BATCH)
        {
            smemA[threadIdx.y][threadIdx.x] = __half2float(A[tid_z*M*K+tid_y*K+i*BLOCK_DIM+threadIdx.x]);
        }
        else
        {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < N && i*BLOCK_DIM+threadIdx.y < K)
        {
            smemB[threadIdx.y][threadIdx.x] = __half2float(B[tid_z*K*N+(i*BLOCK_DIM+threadIdx.y)*N+tid_x]);
        }
        else
        {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for(int j = 0;j < BLOCK_DIM; ++j)
        {
            psum += smemA[threadIdx.y][j]*smemB[j][threadIdx.x];
        }
        __syncthreads();

    }
    if(tid_x < N && tid_y < M && tid_z <BATCH)
    {
        C[global_tid] = __float2half(psum);
    }
    
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) 
{
    dim3 threadsPerBlock(16,16,1);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x,(M+threadsPerBlock.y-1)/threadsPerBlock.y,(BATCH+threadsPerBlock.z-1)/threadsPerBlock.z);
    size_t shared_mem_size = 2*BLOCK_DIM*BLOCK_DIM*sizeof(float);

    matrix_mult_fp16_kernel<<<blocksPerGrid,threadsPerBlock,shared_mem_size>>>(A,B,C,BATCH,M,N,K);
    cudaDeviceSynchronize();
}
