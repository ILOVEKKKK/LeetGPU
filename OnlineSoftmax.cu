#include <cuda_runtime.h>
#include <float.h>
#include <vector>
#include <iostream>
#define MASK_ALL 0xffffffff
#define BLOCK_DIM 128
__device__ inline float warp_reduce_max(float val)
{
    for(int stride = 16;stride > 0;stride >>=1)
    {
        val = fmaxf(val,__shfl_xor_sync(MASK_ALL,val,stride));
    }
    return val;
}

__device__ inline float warp_reduce_sum(float val)
{
    for(int stride = 16;stride > 0;stride >>=1)
    {
        val += __shfl_xor_sync(MASK_ALL,val,stride);
    }
    return val;
}

//单warp内online softmax，blockDim.x = 32
__global__ void online_softmax_kernel(const float* input,float* output,int N)
{
    int tid = threadIdx.x;
    float cur_max = -FLT_MAX;
    float cur_sum = 0.0f;

    for(int i = tid;i < N;i += blockDim.x)
    {
        float val = input[i];
        float max_old = cur_max;

        if(val > max_old)
        {
            cur_max = val;
            cur_sum = cur_sum*expf(max_old - val) + 1.0f;
        }
        else
        {
            cur_sum += expf(val-cur_max);
        }
    }

    float max_val = warp_reduce_max(cur_max);
    
    __shared__ float s_max;
    if(tid == 0)
    {
        s_max = max_val;
    }
    __syncthreads();

    float sum_val = warp_reduce_sum(cur_sum*expf(cur_max-s_max));
    __shared__ float s_sum;
    if(tid == 0)
    {
        s_sum = sum_val;
    }
    __syncthreads();

    for(int i = tid;i < N;i+=blockDim.x)
    {
        output[i] = expf(input[i]-s_max)/s_sum;
    }
}

//多warp版本的online softmax，适用于blockDim.x > 32的情况
__global__ void online_softmax_kernel_multwarp(const float* input,float* output,int N)
{
    int tid= threadIdx.x;
    int lane_id = tid%32;//线程在warp内的id
    int warp_id = tid/32;//warp在block内的id
    int warp_num = blockDim.x/32;//block内warp的数量

    float cur_max = -FLT_MAX;
    float cur_sum = 0.0f;

    //shared_mem存放每个warp的max和sum，大小为warp_num
    __shared__ float smem_max[BLOCK_DIM/32];
    __shared__ float smem_sum[BLOCK_DIM/32];

    for(int i = tid;i < N;i += blockDim.x)
    {
        float cur_val = input[i];
        float old_max = cur_max;
        if(cur_val > old_max)
        {
            cur_max = cur_val;
            cur_sum = cur_sum*expf(old_max-cur_max) + 1.0f;
        }
        else
        {
            cur_sum += expf(cur_val-cur_max);
        }
    }

    float max_val = warp_reduce_max(cur_max);
    float sum_val = warp_reduce_sum(cur_sum*expf(cur_max-max_val));

    //每个warp的第一个线程将当前warp的max和sum写入shared_mem
    if(lane_id == 0)
    {
        smem_max[warp_id] = max_val;
        smem_sum[warp_id] = sum_val;
    }
    __syncthreads();

    //warp间规约得到全局最大max和sum
    for(int i = warp_num/2;i > 0;i >>= 1)
    {
        if(warp_id < i)
        {
            float warp_max = fmaxf(smem_max[warp_id],smem_max[warp_id+i]);
            if(warp_max == smem_max[warp_id])
            {
                smem_sum[warp_id] += smem_sum[warp_id+i]*expf(smem_max[warp_id+i]-warp_max);
            }
            else
            {
                smem_sum[warp_id] = smem_sum[warp_id]*expf(smem_max[warp_id]-warp_max) + smem_sum[warp_id + i];
            }
            smem_max[warp_id] = warp_max;
        }
        __syncthreads();
    }

    float final_max = smem_max[0];
    float final_sum = smem_sum[0];

    for(int i = tid; i < N;i += blockDim.x)
    {
        output[i] = expf(input[i]-final_max)/final_sum;
    }

}

__host__ void check(float* input,int N)
{
    for(int i = 0;i < 128;++i)
    {
        std::cout << "GPU Result:" << input[i] << std::endl;
    }
    return;
}

int main()
{
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid(1);

    int N = 1024;
    std::vector<float>input(N,1.0f);
    std::vector<float>output;
    output.resize(N);

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, N*sizeof(float));
    cudaMalloc(&d_output, N*sizeof(float));
    cudaMemcpy(d_input, input.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    online_softmax_kernel_multwarp<<<blocksPerGrid,threadsPerBlock>>>(d_input,d_output,N);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, N*sizeof(float), cudaMemcpyDeviceToHost);

    check(output.data(),N);

    cudaFree(d_input);
    cudaFree(d_output);
}