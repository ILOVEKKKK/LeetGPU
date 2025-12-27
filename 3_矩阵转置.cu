#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockDim.x+threadIdx.x;
    if(row < rows&&col < cols)
    {
        output[col*rows+row] = input[row*cols+col];//转置的原理就是行优先遍历的顺序换为列优先顺序的遍历
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
