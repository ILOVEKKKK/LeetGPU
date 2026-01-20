#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void matrix_mult_fp16_kernel(const half* A,const half*B,half* C,int BATCH,int M,int N,int K)
{

}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {}
