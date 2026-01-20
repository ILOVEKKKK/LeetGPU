#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>

//逐点向量相乘kernel
__global__ void elementwise_mult_kernel(const float* vec1,const float* vec2,float* output,int M,int N)
{
    int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
    int tid_y = blockIdx.y*blockDim.y+threadIdx.y;
    int global_tid = tid_y * N+tid_x;

    if(tid_x < N && tid_y < M)
    {
        output[global_tid] = vec1[global_tid]*vec2[global_tid];
    }
}

//rotate_half算子，对于一个M*N矩阵，交换矩阵每一行的前一半和后一半，并对交换完的前一半取反
__global__ void rotate_half_kernel(float* input, float* output, int M, int D)
{
    int tid = threadIdx.x;
    int row_id = blockIdx.x;

    if(row_id < M)
    {
        for(int i = tid;i < D/2;++i)
        {
            std::swap(input[row_id*D+tid],input[row_id*D+tid+D/2]);
            input[row_id*D+tid] *= -1;
        }
        output[row_id*D+tid] = input[row_id*D+tid];
    }
}

__global__ void add_kernel(const float* input1,const float* input2,float* output,int M,int D)
{
    int tid_x = threadIdx.x+blockIdx.x*blockDim.x;
    int tid_y = threadIdx.y+blockIdx.y*blockDim.y;
    int global_tid = tid_y * D+tid_x;

    if(tid_x < D && tid_y < M)
    {
        output[global_tid] = input1[global_tid]+input2[global_tid];
    }
}

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) 
{
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((D+threadsPerBlock.x-1)/threadsPerBlock.x,(M+threadsPerBlock.y-1)/threadsPerBlock.y);
    float* x_cos = NULL;
    cudaMalloc(&x_cos,M*D*sizeof(float));
    elementwise_mult_kernel<<<blocksPerGrid,threadsPerBlock>>>(Q,cos,x_cos,M,D);
    cudaDeviceSynchronize();

    float* rotate_x = NULL;
    cudaMalloc(&rotate_x,M*D*sizeof(float));
    rotate_half_kernel<<<blocksPerGrid,threadsPerBlock>>>(Q,rotate_x,M,D);
    cudaDeviceSynchronize();

    float* rotate_sin = NULL;
    cudaMalloc(&rotate_sin,M*D*sizeof(float));
    elementwise_mult_kernel<<<blocksPerGrid,threadsPerBlock>>>(rotate_x,sin,rotate_sin,M,D);
    cudaDeviceSynchronize();

    add_kernel<<<blocksPerGrid,threadsPerBlock>>>(x_cos,rotate_sin,output,M,D);
    cudaDeviceSynchronize();

    cudaFree(x_cos);
    cudaFree(rotate_x);
    cudaFree(rotate_sin);
}
