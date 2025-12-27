#include <cuda_runtime.h>

//Softmax算子，按照行优先进行
__global__ void rowwise_softmax_kernel(const float * input,float* output,int M,int N)
{
    //一个block对应一行
    int row = blockIdx.x;
    if(row >= M) return;
    extern __shared__ float share[];

    //按行优先顺序计算输入矩阵中每一行的最大值
    float local_max = -1e10;
    for(int i = threadIdx.x;i<N;i+=blockDim.x)
    {
        local_max = fmaxf(local_max,input[row*N+i])
    }
    share[threadIdx.x] = local_max;
    __syncthreads();

    for(int s = blockDim.x/2;s>0;s/=2)
    {
        if(threadIdx.x < s)
        {
            share[s] = fmaxf(share[s],share[threadIdx.x+s]);
        }
        __syncthreads();
    }
    float row_max = share[0];

    //按行优先顺序计算输入矩阵中每一行的总和
    float local_sum = 0;
    for(int i = threadIdx.x;i<N;i+=blockDim.x)
    {
        local_sum += expf(input[row*N+i]-row_max);
    }
    share[threadIdx.x] = local_sum;
    __syncthreads();

    for(int s = blockDim.x/2;s>0;s/=2)
    {
        if(threadIdx.x < s)
        {
            share[threadIdx.x] += share[threadIdx.x+s];
        }
        __syncthreads();
    }
    float row_sum = share[0];

    //按照行优先顺序填充输出矩阵中每一个位置的softmax结果
    for(int i = threadIdx.x;i<N;i+=blockDim.x)
    {
        output[row*N+i] = expf(input[row*N+i]-row_max)/row_sum;
    }
}

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

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockDim.x+threadIdx.x;
    if(row < rows&&col < cols)
    {
        output[row*rows+col] = input[col*cols+row];//转置的原理就是行优先遍历的顺序换为列优先顺序的遍历
    }
}

__global__ void times_aray(float* array,int k,int N)
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < N)
    {
        array[idx]*=k;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) 
{
    float* KT;
    cudaMalloc(&KT,N*d*sizeof(float));
    dim3 BlockDim(16,16);
    matrix_transpose_kernel<<<dim3((N+BlockDim.x-1)/BlockDim.x,(d+BlockDim.y-1)/BlockDim.y),BlockDim>>>(K,KT,N,d);
    
    float* QKT,sQKT*;
    cudaMalloc(&QKT,M*N*sizeof(float));
    cudaMalloc(&sQKT,M*N*sizeof(float));

    matrix_multiplication_kernel<<<dim3((N+BlockDim.x-1)/BlockDim.x,(M+BlockDim.y-1)/BlockDim.y),BlockDim>>>(Q,KT,QKT,M,d,N);
    times_array<<<(M*N+256-1)/256,256>>>(QKT,sqrt(d),M*N);
    row_softmax_kernel(QKT,sQKT,M,N);
    matrix_multiplication_kernel<<<dim3((d+BlockDim.x-1)/BlockDim.x,(M+BlockDim.y-1)/BlockDim.y),BlockDim>>>(sQKT,V,output,M,d,N);

    cudaDeviceSynchronize();
    cudaFree(KT);
    cudaFree(QKT);
    cudaFree(sQKT);

}
