#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void matrix_mult_kernel(const half* A, const half* B,half* C,int M,int N,int K,float alpha,float beta)
{
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*N+ix;
    if(ix < N&&iy < M)
    {
        float sum = 0.0f;
        for(int i = 0;i < K;i++)
        {
            float a_val = __half2float(A[iy*K+i]);
            float b_val = __half2float(B[i*N+ix]);
            sum += a_val*b_val;
        }
        float old_c = __half2float(C[idx]);
        float result = alpha*sum+beta*old_c;
        C[idx] = __float2half(result);
    }
}


// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) 
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_mult_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K,alpha,beta);
    cudaDeviceSynchronize();
}
