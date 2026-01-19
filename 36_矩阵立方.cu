#include <cuda_runtime.h>

__global__ void matrix_mult_kernel(const float* input1, const float*input2,float* output, int N)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y*N+tid_x;
    if(tid_x < N && tid_y < N)
    {
        float psum = 0.0f;
        for(int i =0;i<N;++i)
        {
            psum += input1[tid_y*N+i]*input2[i*N+tid_x];
        }
        output[global_tid] = psum;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P)
{
    if (P == 1) {
        cudaMemcpy(output, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        return;
    }
    dim3 ThreadsPerBlock(16,16);
    dim3 BlocksPerGrid((N+ThreadsPerBlock.x-1)/ThreadsPerBlock.x,(N+ThreadsPerBlock.y-1)/ThreadsPerBlock.y);
    if(P == 2)
    {
        matrix_mult_kernel<<<BlocksPerGrid,ThreadsPerBlock>>>(input,input,output,N);
        cudaDeviceSynchronize();
        return;
    }
    float* tempbuffer = NULL;
    cudaMalloc(&tempbuffer,N*N*sizeof(float));
    matrix_mult_kernel<<<BlocksPerGrid,ThreadsPerBlock>>>(input,input,tempbuffer,N);
    for(int i = 0;i<P-2;++i)
    {
        matrix_mult_kernel<<<BlocksPerGrid,ThreadsPerBlock>>>(input,tempbuffer,tempbuffer,N);
    }
    cudaMemcpy(output,tempbuffer,N*N*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaFree(tempbuffer);
    cudaDeviceSynchronize();
}