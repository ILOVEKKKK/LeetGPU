#include <cuda_runtime.h>
#include <float.h>
#define BLOCK_DIM 16

//归约求最大值，求一个M*N矩阵的每一行最大值，输出N*1向量
__global__ void max_reduction_kernel(const float* input,int M,int N,float* output)
{
    //让一个block负责一行
    int tid = threadIdx.x;
    int row_id = blockIdx.x;
    
    if(row_id >= M) return;

    float pmax = -FLT_MAX;
    extern __shared__ float smem[];
    
    //跨步循环，避免一个线程负责一个元素时共享内存的溢出
    for(int i = tid;i < N;i += blockDim.x)
    {
        pmax = fmaxf(pmax,input[row_id*N+i]);
    }

    //shared_mem内存有限，为了防止溢出，需要使用跨步循环，一个线程访问同一行间隔为blockDim.x的多个元素并取出最大值
    smem[tid] = pmax;
    __syncthreads();

    for(int i = blockDim.x/2;i > 0;i >>= 1)
    {
        if(tid < i)
        {
            smem[tid] = fmaxf(smem[tid],smem[tid+i]);
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        output[row_id] = smem[0];
    }
}

//归约求指数和，求一个M*N矩阵的每一行和，输出N*1向量
__global__ void add_reduction_kernel(const float* input,const float* row_max,int M,int N,float* output)
{
    int tid = threadIdx.x;
    int row_id = blockIdx.x;

    if(row_id >= M) return;
    
    extern __shared__ float smem[];
    float psum = 0.0f;

    for(int i = tid;i<N;i += blockDim.x)
    {
        psum += expf(input[row_id*N+i]-row_max[row_id]);
    }
    
    smem[tid] = psum;
    __syncthreads();

    for(int i = blockDim.x/2;i >0;i >>=1)
    {
        if(tid < i)
        {
            smem[tid] += smem[tid+i];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        output[row_id] = smem[0];
    }
}

//对M*N矩阵按行进行softmax计算，输出一个M*N矩阵
__global__ void softmax_kernel(const float* input,int M,int N,const float* row_max,const float* row_sum,float* output)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y*N+tid_x;

    if(tid_x < N && tid_y < M)
    {
        output[global_tid] = expf(input[global_tid]-row_max[tid_y])*(1.0f/(row_sum[tid_y]));
    }
}

//矩阵转置kernel
__global__ void matrix_transpose_kernel(const float* input,float* output,int M,int N)
{
    __shared__ float smem[BLOCK_DIM][BLOCK_DIM];
    
    int block_x = blockIdx.x*BLOCK_DIM;
    int block_y = blockIdx.y*BLOCK_DIM;

    int tid_x = block_x+threadIdx.x;
    int tid_y = block_y+threadIdx.y;

    if(tid_x < N && tid_y < M)
    {
        smem[threadIdx.y][threadIdx.x] = input[tid_y*N+tid_x];
    }
    __syncthreads();

    int x_out = blockIdx.y*BLOCK_DIM+threadIdx.x;
    int y_out = blockIdx.x*BLOCK_DIM+threadIdx.y;

    if(x_out < M && y_out <N)
    {
        output[y_out*M+x_out] = smem[threadIdx.x][threadIdx.y];
    }
}

//矩阵乘法kernel
__global__ void matrix_mult_kernel(const float* input1,const float* input2,int M,int N,int K,float* output)
{
    int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
    int tid_y = blockIdx.y*blockDim.y+threadIdx.y;
    int global_tid = tid_y*N+tid_x;

    __shared__ float smemA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float smemB[BLOCK_DIM][BLOCK_DIM];

    float psum = 0.0f;
    for(int i=0; i<(K+BLOCK_DIM-1)/BLOCK_DIM; ++i)
    {
        if(tid_y < M && i*BLOCK_DIM+threadIdx.x < K)
        {
            smemA[threadIdx.y][threadIdx.x] = input1[tid_y * K+i*BLOCK_DIM+threadIdx.x];
        }
        else
        {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < N && i*BLOCK_DIM+threadIdx.y < K)
        {
            smemB[threadIdx.y][threadIdx.x] = input2[(i*BLOCK_DIM+threadIdx.y)*N+tid_x];
        }
        else
        {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        
        for(int i = 0;i< BLOCK_DIM;++i)
        {
            psum += smemA[threadIdx.y][i]*smemB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if(tid_x < N && tid_y < M)
    {
        output[global_tid] = psum;
    }
}

//计算attention bias的kernel
__global__ void attention_bias_kernel(const float* input,int d,float alpha,int M,int N,float* output)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y * N+tid_x;

    if(tid_x < N && tid_y < M)
    {
        output[global_tid] = (input[global_tid]/sqrtf(d))+alpha*(tid_y-tid_x);
    }
}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) 
{
    //step1:对矩阵K进行转置
    dim3 threadsPerBlock(BLOCK_DIM,BLOCK_DIM);
    dim3 blocksPerGrid((d+threadsPerBlock.x-1)/threadsPerBlock.x,(N+threadsPerBlock.y-1)/threadsPerBlock.y);
    float* K_T = NULL;
    cudaMalloc(&K_T,N*d*sizeof(float));
    matrix_transpose_kernel<<<blocksPerGrid,threadsPerBlock>>>(K,K_T,N,d);
    cudaDeviceSynchronize();

    //step2:计算QKT
    blocksPerGrid.x = (N+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (M+threadsPerBlock.y-1)/threadsPerBlock.y;
    float* Q_K_T = NULL;
    cudaMalloc(&Q_K_T,M*N*sizeof(float));
    size_t shared_mem_size = 2*BLOCK_DIM*BLOCK_DIM*sizeof(float);
    matrix_mult_kernel<<<blocksPerGrid,threadsPerBlock>>>(Q,K_T,M,N,d,Q_K_T);
    cudaDeviceSynchronize();

    //step3:计算attention_bias
    float* attention_bias = NULL;
    cudaMalloc(&attention_bias,M*N*sizeof(float));
    attention_bias_kernel<<<blocksPerGrid,threadsPerBlock>>>(Q_K_T,d,alpha,M,N,attention_bias);
    cudaDeviceSynchronize();

    //step4:计算softmax，先求最大值，求和，最后再就softmax结果
    dim3 threadsPerBlock_Softmax(256);
    dim3 blocksPerGrid_Softmax(M);
    float* row_max =NULL;
    float* row_sum =NULL;
    float* softmax_result = NULL;
    shared_mem_size = threadsPerBlock_Softmax.x*sizeof(float);
    cudaMalloc(&row_max,M*sizeof(float));
    cudaMalloc(&row_sum,M*sizeof(float));
    cudaMalloc(&softmax_result,M*N*sizeof(float));
    max_reduction_kernel<<<blocksPerGrid_Softmax,threadsPerBlock_Softmax,shared_mem_size>>>(attention_bias,M,N,row_max);
    cudaDeviceSynchronize();
    add_reduction_kernel<<<blocksPerGrid_Softmax,threadsPerBlock_Softmax,shared_mem_size>>>(attention_bias,row_max,M,N,row_sum);
    cudaDeviceSynchronize();
    dim3 new_threadsPerBlock(16,16);
    dim3 new_blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x,(M+threadsPerBlock.y-1)/threadsPerBlock.y);
    softmax_kernel<<<new_blocksPerGrid,new_threadsPerBlock>>>(attention_bias,M,N,row_max,row_sum,softmax_result);
    cudaDeviceSynchronize();

    //step5:计算和V的乘积
    blocksPerGrid.x = (d+threadsPerBlock.x-1)/threadsPerBlock.x;
    blocksPerGrid.y = (M+threadsPerBlock.y-1)/threadsPerBlock.y;
    shared_mem_size = 2*BLOCK_DIM*BLOCK_DIM*sizeof(float);
    matrix_mult_kernel<<<blocksPerGrid,threadsPerBlock,shared_mem_size>>>(softmax_result,V,M,d,N,output);
    cudaDeviceSynchronize();

    cudaFree(K_T);
    cudaFree(Q_K_T);
    cudaFree(attention_bias);
    cudaFree(row_sum);
    cudaFree(row_max);
    cudaFree(softmax_result);
}
