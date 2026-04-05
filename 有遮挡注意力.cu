#include <cuda_runtime.h>
#include <float.h>
#define BLOCKDIM 16

__global__ void KT_transpose_kernel(const float* K,float* output,int M,int d)
{
    __shared__ float smem[BLOCKDIM][BLOCKDIM];
    
    int block_x = blockIdx.x*BLOCKDIM;
    int block_y = blockIdx.y*BLOCKDIM;

    int tid_x = block_x+threadIdx.x;
    int tid_y = block_y+threadIdx.y;

    if(tid_x < d && tid_y < M)
    {
        smem[threadIdx.y][threadIdx.x] = K[tid_y*d+tid_x];
    }
    __syncthreads();

    int x_out = blockIdx.y*BLOCKDIM+threadIdx.x;
    int y_out = blockIdx.x*BLOCKDIM+threadIdx.y;

    if(x_out < M && y_out <d)
    {
        output[y_out*M+x_out] = smem[threadIdx.x][threadIdx.y];
    }
}

__global__ void QKT_mult_kernel(const float* Q,const float* KT,float* output,int M,int d)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    __shared__ float smemA[BLOCKDIM][BLOCKDIM];
    __shared__ float smemB[BLOCKDIM][BLOCKDIM];
    float psum = 0.0f;

    for(int i = 0;i<(d+BLOCKDIM-1)/BLOCKDIM;++i)
    {
        if(tid_y < M && i*BLOCKDIM+threadIdx.x < d)
        {
            smemA[threadIdx.y][threadIdx.x] = Q[tid_y*d+i*BLOCKDIM+threadIdx.x];
        }
        else
        {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < M && i*BLOCKDIM+threadIdx.y < d)
        {
            smemB[threadIdx.y][threadIdx.x] = KT[(i*BLOCKDIM+threadIdx.y)*M+tid_x];
        }
        else
        {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for(int j = 0;j < BLOCKDIM;++j)
        {
            psum += smemA[threadIdx.y][j]*smemB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(tid_x < M && tid_y < M)
    {
        output[tid_y*M+tid_x] = psum/sqrtf((float)d);
    }
    
}

__global__ void masked_kernel(const float* QKT,float* output,int M)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y*M+tid_x;

    if(tid_x < M && tid_y < M)
    {
        output[global_tid] = (tid_x <= tid_y)?QKT[global_tid]:-FLT_MAX;
    }
}

__global__ void add_reduction_kernel(const float* input,const float* row_max,float*output,int M)
{
    int tid_x = threadIdx.x;
    int row_id = blockIdx.x;
    extern __shared__ float smem[];
    float psum = 0.0f;
    if(row_id < M)
    {
        for(int i = tid_x;i < M;i += blockDim.x)
        {
            psum += expf(input[row_id*M+i]-row_max[row_id]); 
        }
        smem[tid_x] = psum;
        __syncthreads();

        for(int i = blockDim.x/2;i > 0;i >>= 1)
        {
            if(tid_x < i)
            {
                smem[tid_x] += smem[tid_x + i];
            }
            __syncthreads();
        }

        if(tid_x == 0)
        {
            output[row_id] = smem[0];
        }
    }
}

__global__ void max_reduction_kernel(const float* input,float*output,int M)
{
    int tid = threadIdx.x;
    int row_id = blockIdx.x;
    extern __shared__ float smem[];
    float pmax = -FLT_MAX;

    for(int i = tid;i < M;i += blockDim.x)
    {
        pmax = fmaxf(pmax,input[row_id*M+i]);
    }
    smem[tid] = pmax;
    __syncthreads();

    for(int i = blockDim.x/2;i > 0;i>>=1)
    {
        if(tid<i)
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

__global__ void softmax_kernel(const float* input,const float* row_sum,const float* row_max,int M,float* output)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int global_tid = tid_y * M+tid_x;

    if(tid_x < M && tid_y < M)
    {
        output[global_tid] = expf(input[global_tid]-row_max[tid_y])/row_sum[tid_y];
    }
}

__global__ void V_mult_kernel(const float* final_QKT,const float* V,int M,int d,float* output)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;

    __shared__ float smemA[BLOCKDIM][BLOCKDIM];
    __shared__ float smemB[BLOCKDIM][BLOCKDIM];

    float psum = 0.0f;

    for(int i = 0;i<(M+BLOCKDIM-1)/BLOCKDIM;++i)
    {
        if(tid_y < M && i*BLOCKDIM+threadIdx.x< M)
        {
            smemA[threadIdx.y][threadIdx.x] = final_QKT[tid_y*M+i*BLOCKDIM+threadIdx.x];
        }
        else
        {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < d && i*BLOCKDIM+threadIdx.y < M)
        {
            smemB[threadIdx.y][threadIdx.x] = V[(i*BLOCKDIM+threadIdx.y)*d+tid_x];
        }
        else
        {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for(int j = 0;j<BLOCKDIM;++j)
        {
            psum += smemA[threadIdx.y][j]*smemB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(tid_x < d && tid_y < M)
    {
        output[tid_y*d+tid_x] = psum;
    } 
}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) 
{
    dim3 threadsPerBlock(BLOCKDIM,BLOCKDIM);
    dim3 blocksPerGrid((M+BLOCKDIM-1)/BLOCKDIM,(d+BLOCKDIM-1)/BLOCKDIM);
    
    float* KT = NULL;
    cudaMalloc(&KT,M*d*sizeof(float));
    size_t smem_size = BLOCKDIM*BLOCKDIM*sizeof(float);
    KT_transpose_kernel<<<blocksPerGrid,threadsPerBlock,0>>>(K,KT,M,d);
    cudaDeviceSynchronize();

    blocksPerGrid = dim3((M+BLOCKDIM-1)/BLOCKDIM,(M+BLOCKDIM-1)/BLOCKDIM);
    float*QKT = NULL;
    cudaMalloc(&QKT,M*M*sizeof(float));
    QKT_mult_kernel<<<blocksPerGrid,threadsPerBlock,0>>>(Q,KT,QKT,M,d);
    cudaDeviceSynchronize();

    float* masked = NULL;
    cudaMalloc(&masked,M*M*sizeof(float));
    masked_kernel<<<blocksPerGrid,threadsPerBlock>>>(QKT,masked,M);
    cudaDeviceSynchronize();

    float* row_sum = NULL;
    float* row_max = NULL;
    cudaMalloc(&row_sum,M*sizeof(float));
    cudaMalloc(&row_max,M*sizeof(float));
    int threads = 256;
    int blocks = M;
    smem_size = threads*sizeof(float);
    max_reduction_kernel<<<blocks,threads,smem_size>>>(masked,row_max,M);
    cudaDeviceSynchronize();
    add_reduction_kernel<<<blocks,threads,smem_size>>>(masked,row_max,row_sum,M);
    cudaDeviceSynchronize();

    float* softmax = NULL;
    cudaMalloc(&softmax,M*M*sizeof(float));
    softmax_kernel<<<blocksPerGrid,threadsPerBlock>>>(masked,row_sum,row_max,M,softmax);
    cudaDeviceSynchronize();

    blocksPerGrid = dim3((d+BLOCKDIM-1)/BLOCKDIM,(M+BLOCKDIM-1)/BLOCKDIM);
    smem_size = 2*BLOCKDIM*BLOCKDIM*sizeof(float);
    V_mult_kernel<<<blocksPerGrid,threadsPerBlock,0>>>(softmax,V,M,d,output);
    cudaDeviceSynchronize();

    cudaFree(KT);
    cudaFree(QKT);
    cudaFree(masked);
    cudaFree(row_max);
    cudaFree(row_sum);
    cudaFree(softmax);
}
