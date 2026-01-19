#include <cuda_runtime.h>
#define TILE 256
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    if (row < M && col < K)
    {
        float sum = 0;
        for (int i = 0; i < N; i++)
        {
            if (row < M && col < K)
            {
                sum += A[row * N + i] * B[i * K + col];
            }
        }
        C[row*K+col] = sum;
    }
}

template<int BLOCK_DIM>
__global__ void matrix_mult_smem_kernel(const float* A,const float*B,float* C,int M,int N,int K)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    float psum = 0.0f;
    __shared__ float smemA[BLOCK_DIM][BLOCKDIM];
    __shared__ float smemB[BLOCK_DIM][BLOCKDIM];

    int TILE_K = (K+BLOCK_DIM-1)/BLOCK_DIM;

    for(int i = 0;i < TILE_K;++i)
    {
        //每次加载一个A矩阵的子矩阵与B矩阵的子矩阵进入shared mem，并进行一次矩阵乘法，结果累加到psum上
        //当该线程负责位置的所有子矩阵都完成计算后，得到了最终结果
        if(tid_y < M && BLOCK_DIM*i+threadIdx.x < K)
        {
            smemA[threadIdx.y][threadIdx.x] = A[tid_y*K+BLOCK_DIM*i+threadIdx.x];
        }
        else
        {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < N && BLOCK_DIM*i+threadIdx.y < K)
        {
            smemB[threadIdx.y][threadIdx.x] = B[(BLOCK_DIM*i+threadIdx.y)*N+tid_x];
        }
        else
        {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for(int s = 0;s < BLOCK_DIM;s++)
        {
            psum+= smemA[threadIdx.y][s]*smemB[s][threadIdx.x];
        }
        __syncthreads();
    }
    if(tid_x < N && tid_y < M)
    {
        C[tid_y*N+tid_x] = psum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}