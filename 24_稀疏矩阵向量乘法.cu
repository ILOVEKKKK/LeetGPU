#include <cuda_runtime.h>

__global__ void sparse_kernel(const float* A, const float* x, float* y, int M, int N, int nnz)
{
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*N+ix;
    if(ix < N && iy < M && A[idx]!=0)
    {
        atomicAdd(&y[iy],x[ix]*A[idx]);
    }

}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) 
{
    dim3 threads(16,16);
    dim3 blocks((N+threads.x-1)/threads.x,(M+threads.y-1)/threads.y);
    sparse_kernel<<<blocks,threads>>>(A,x,y,M,N,nnz);
    cudaDeviceSynchronize();
}