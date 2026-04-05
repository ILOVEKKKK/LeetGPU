#include <cuda_runtime.h>
#define BLOCKDIM 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockDim.x+threadIdx.x;
    if(row < rows&&col < cols)
    {
        output[col*rows+row] = input[row*cols+col];//转置的原理就是行优先遍历的顺序换为列优先顺序的遍历
    }
}

__global__ void matrix_transpose_share_kernel(const float* input, float* output, int M ,int N)
{
    int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
    int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

    __shared__ float smem[BLOCKDIM][BLOCKDIM];

    if(tid_x < N && tid_y < M)
    {
        smem[threadIdx.y][threadIdx.x] = input[tid_y*N+tid_x];
        __syncthreads();
    }

    int x_out = blockDim.y*blockIdx.y+threadIdx.x;
    int y_out = blockDim.x*blockIdx.x+threadIdx.y;

    if(x_out < M && y_out < N)
    {
        output[y_out * M+x_out] = smem[threadIdx.x][threadIdx.y];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
