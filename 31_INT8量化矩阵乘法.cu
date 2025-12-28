#include <cuda_runtime.h>

__global__ void INT8_matrix_mult_kernel(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C)
{
    int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
    int tid_y = blockIdx.y*blockDim.y+threadIdx.y;
    int global_tid = tid_y*N+tid_x;

    if(tid_x < N && tid_y < M)
    {
        int32_t sum = 0.0f;
        for(int i = 0;i < K;++i)
        {
            int32_t a = (int32_t)A[tid_y*K+i];
            int32_t b = (int32_t)B[i*N+tid_x];
            sum += (a-zero_point_A)*(b-zero_point_B); 
        }
        float scaled_sum = ((float)sum*scale_A*scale_B)/scale_C;
        float final_sum = roundf(scaled_sum)+(float)zero_point_C;
        if(final_sum < -128.0f) final_sum = -128.0f;
        if(final_sum > 127.0f) final_sum = 127.0f;
        C[global_tid] = (int8_t)final_sum;
    }

}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) 
{
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x,(M+threadsPerBlock.y-1)/threadsPerBlock.y);
    INT8_matrix_mult_kernel<<<blocksPerGrid,threadsPerBlock>>>(A,B,C,M,N,K,scale_A,scale_B,scale_C,zero_point_A, zero_point_B, zero_point_C);
    cudaDeviceSynchronize();
}
