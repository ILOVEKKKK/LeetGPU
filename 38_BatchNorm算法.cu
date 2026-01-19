#include <cuda_runtime.h>

//按列求每一列的均值与方差
__global__ void sum_var_kernel(const float* input,float* output_sum,float* output_var,int N,int C)
{
    int tid_y = threadIdx.y;
    int tid_x = blockIdx.x;

    extern __shared__ float smem[];
    float* smem_sum = smem;
    float* smem_var = &smem[blockDim.y];
   
    float psum = 0.0f;
    float pvar = 0.0f;
    //以blockDim.y为跨度，每一个线程累加部分和，则所有block的线程再累加就是整列和
    for(int i = tid_y;i < N;i += blockDim.y)
    {
        float val = input[i*C+blockIdx.x];
        psum += val;
        pvar += val*val;
    }

    smem_sum[tid_y] = psum;
    smem_var[tid_y] = pvar;
    __syncthreads();
    
    //规约求和得到当前block内所有线程求得的该列部分和
    for(int i = blockDim.y/2; i>0 ;i >>=1)
    {
        if(tid_y < i)
        {
            smem_sum[tid_y] += smem_sum[tid_y+i];
            smem_var[tid_y] += smem_var[tid_y+i];
        }
        __syncthreads();
    }

    if(tid_y == 0)
    {
        atomicAdd(&output_sum[blockIdx.x],smem_sum[0]);
        atomicAdd(&output_var[blockIdx.x],smem_var[0]);
    }
}

__global__ void mean_std_kernel(const float* output_sum,const float* output_var,float* mean,float* std,int N,int C)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    if(tid_x < C)
    {
        float cur_mean = output_sum[tid_x]/N;
        mean[tid_x] = cur_mean;
        std[tid_x] = (output_var[tid_x]/N)-cur_mean*cur_mean;
    }
}

__global__ void batchnorm_kernel(const float* input, const float* gamma, const float* beta, const float* mean,const float* std,float* output,
                      int N, int C, float eps)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y*C+tid_x;

    if(tid_x < C && tid_y < N)
    {
        float cur_x = (input[global_tid]-mean[tid_x])/sqrtf(std[tid_x]+eps);
        float cur_y = gamma[tid_x]*cur_x+beta[tid_x];
        output[global_tid] = cur_y;
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) 
{
    dim3 threadsPerBlock(1,256);
    dim3 blocksPerGrid(C,1);
    size_t shared_mem_size = 2*threadsPerBlock.y*sizeof(float);
    
    float* output_sum = NULL;
    float* output_var = NULL;
    float* mean = NULL;
    float* std = NULL;
    cudaMalloc(&output_sum,C*sizeof(float));
    cudaMalloc(&output_var,C*sizeof(float));
    cudaMalloc(&mean,C*sizeof(float));
    cudaMalloc(&std,C*sizeof(float));


    sum_var_kernel<<<blocksPerGrid,threadsPerBlock,shared_mem_size>>>(input,output_sum,output_var,N,C);
    cudaDeviceSynchronize();

    mean_std_kernel<<<blocksPerGrid,threadsPerBlock>>>(output_sum,output_var,mean,std,N,C);
    cudaDeviceSynchronize();

    dim3 new_threadsPerBlock(16,16);
    dim3 new_blocksPerGrid((C+new_threadsPerBlock.x-1)/new_threadsPerBlock.x,(N+new_threadsPerBlock.y-1)/new_threadsPerBlock.y);
    batchnorm_kernel<<<new_blocksPerGrid,new_threadsPerBlock>>>(input,gamma,beta,mean,std,output,N,C,eps);
    cudaDeviceSynchronize();
    
    cudaFree(output_var);
    cudaFree(output_sum);
    cudaFree(mean);
    cudaFree(std);
    
}
