#include <cuda_runtime.h>

__global__ void count_3d_array_element_kernel(const int* input, int* output, int N, int M, int K, int P)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int tid_z = blockDim.z*blockIdx.z+threadIdx.z;
    int global_tid = tid_z*N*M+tid_y*M+tid_x;

    if(tid_x < M && tid_y < N && tid_z < K)
    {
        if(input[global_tid] == P)
        {
            atomicAdd(output,1);
        }
    }
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) 
{
    dim3 threadsPerBlock(8,8,8);
    dim3 blocksPerGrid((M+threadsPerBlock.x-1)/threadsPerBlock.x,(N+threadsPerBlock.y-1)/threadsPerBlock.y,(K+threadsPerBlock.z-1)/threadsPerBlock.z);
    count_3d_array_element_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,output,N,M,K,P);
    cudaDeviceSynchronize();
}
