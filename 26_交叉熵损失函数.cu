#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* logits, const int* true_labels, float* loss, int N, int C) 
{
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;
    int global_tid = C*blockIdx.x+threadIdx.x;

    smem[tid] = (tid < C)?expf(logits[global_tid]):0.0f;
    __syncthreads();

    int index = true_labels[blockIdx.x];
    float yj = logits[C*blockIdx.x+index];
    
    for(int i = blockDim.x/2; i > 0;i>>=1)
    {
        if(tid < i)
        {
            smem[tid] += smem[tid+i];
        }
        __syncthreads();
    }

    float log_sum = __logf(smem[0]);
    __syncthreads();
    
    if(tid == 0)
    {
        atomicAdd(loss,(log_sum-yj)/N);
    }
}

int nextPow2(int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) 
{
    int threadsperblock = nextPow2(C);
    int blockspergrid = N;
    size_t shared_mem_size = threadsperblock*sizeof(float);
    cross_entropy_kernel<<<blockspergrid,threadsperblock,shared_mem_size>>>(logits, true_labels, loss, N, C);
    cudaDeviceSynchronize();
}
