#include <cuda_runtime.h>


__global__ void conv_kernel(const float* input,const float* kernel,float* output,int input_rows,int kernel_rows,int output_rows,int input_cols,int kernel_cols,int output_cols)
{
    int output_row = blockDim.y*blockIdx.y+threadIdx.y;
    int output_col = blockDim.x*blockIdx.x+threadIdx.x;
    if(output_row < output_rows&&output_col<output_cols)
    {
        float sum = 0.0f;
        for(int i =0;i<kernel_rows;i++)
        {
            for(int j =0;j<kernel_cols;j++)
            {
                int num_row = output_row+i;
                int num_col = output_col+j;
                sum += input[num_row*input_cols+num_col]*kernel[i*kernel_cols+j];
            }
        }
        output[output_row*output_cols+output_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) 
{
    int output_rows = input_rows-kernel_rows+1;
    int output_cols = input_cols-kernel_cols+1;
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((output_rows+threadsPerBlock.x+1)/threadsPerBlock.x,(output_cols+threadsPerBlock.y+1)/threadsPerBlock.y);
    conv_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,kernel,output,input_rows,kernel_rows,output_rows,input_cols,kernel_cols,output_cols);
    cudaDeviceSynchronize();
}