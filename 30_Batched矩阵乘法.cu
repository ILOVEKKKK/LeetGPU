#include <cuda_runtime.h>

__global__ void matrix_mult_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int batch_id = blockIdx.z;
    int global_tid = M*N*batch_id+tid_y*N+tid_x;

    if(tid_x < N && tid_y < M && batch_id < BATCH)
    {
        float sum = 0.0f;
        for(int i = 0; i < K; ++i)
        {
            float a = A[M*K*batch_id+tid_y*K+i];
            float b = B[K*N*batch_id+i*N+tid_x];
            sum += a*b;
        }
        C[global_tid] = sum;
    }
}
// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) 
{
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x,(M+threadsPerBlock.y-1)/threadsPerBlock.y,BATCH);
    matrix_mult_kernel<<<blocksPerGrid,threadsPerBlock>>>(A,B,C,BATCH,M,N,K);
    cudaDeviceSynchronize();
}
