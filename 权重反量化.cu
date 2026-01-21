#include <cuda_runtime.h>

__global__ void weight_dequant_kernel(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y * N+tid_x;

    if(tid_x < N && tid_y < M)
    {
        int S_x = tid_x / TILE_SIZE;
        int S_y = tid_y / TILE_SIZE;
        int M_S = (M+TILE_SIZE-1) / TILE_SIZE;
        int N_S = (N+TILE_SIZE-1) / TILE_SIZE;
        Y[global_tid] = X[global_tid]*S[S_y*N_S+S_x];
    }
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) 
{
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x,(M+threadsPerBlock.y-1)/threadsPerBlock.y);
    weight_dequant_kernel<<<blocksPerGrid,threadsPerBlock>>>(X,S,Y,M,N,TILE_SIZE);
    cudaDeviceSynchronize();
}