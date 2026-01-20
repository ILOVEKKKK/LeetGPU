#include <cuda_runtime.h>
#include <float.h>

__global__ void max_kernel(const float* input,float* result,int N)
{
  int tid = threadIdx.x;
  int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
  extern __shared__ float smem[];
  
  float max_val = -FLT_MAX;
  for(int i = global_tid;i < N;i += gridDim.x*blockDim.x)
  {
    max_val = fmaxf(input[global_tid],max_val);
  }

  smem[tid] = max_val;
  __syncthreads();

  for(int i = blockDim.x/2;i > 0; i >>= 1)
  {
    if(tid < i)
    {
      smem[tid] += fmaxf(smem[tid],smem[tid+i]);
    }
    __syncthreads();
  }

  if(tid == 0)
  {
    atomicMax(result,smem[0]);
  }
  

}

__global__ void softmax_kernel(const float* input, float* output, int N,float* max_val,float* sum)
{
    int tid = threadIdx.x;
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
    extern __shared__ float smem[];
    smem[tid] = global_tid < N?expf(input[global_tid]-max_val[0]):0.0f;
    __syncthreads();

    for(int i = blockDim.x/2;i > 0;i >>= 1)
    {
      if(tid < i)
      {
        smem[tid] += smem[tid+i];
      }
      __syncthreads();
    }

    if(tid == 0)
    {
      atomicAdd(sum,smem[0]);
    }

    if(global_tid < N)
    {
      output[global_tid] = expf(input[global_tid])*(1.0f/sum[0]);
    }
}
    
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* result = NULL;
    float* sum = NULL;
    cudaMalloc(&result,sizeof(float));
    cudaMalloc(&sum,sizeof(float));

    size_t shared_mem_size = threadsPerBlock*sizeof(float);
    max_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,result,N);
    cudaDeviceSynchronize();
    
    softmax_kernel<<<blocksPerGrid, threadsPerBlock,shared_mem_size>>>(input, output, N,result,sum);
    cudaDeviceSynchronize();


}
