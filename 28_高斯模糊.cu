#include <cuda_runtime.h>

__global__ void gaussian_blur_kernel(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;

    if(tid_x >= 0 && tid_x < input_cols&&tid_y >= 0 && tid_y < input_rows)
    {
        float sum = 0.0f;
        
        int half_kernel_row = kernel_rows/2;
        int half_kernel_col = kernel_cols/2;

        for(int kernel_row = 0;kernel_row < kernel_rows;++kernel_row)
        {
            for(int kernel_col = 0;kernel_col < kernel_cols;++kernel_col)
            {
                int cur_pixel_x = tid_x + kernel_col - half_kernel_col;
                int cur_pixel_y = tid_y + kernel_row - half_kernel_row;
                if(cur_pixel_x >=0 && cur_pixel_x < input_cols && cur_pixel_y >=0 && cur_pixel_y < input_rows)
                {
                    float cur_pixel_val = input[cur_pixel_y*input_cols+cur_pixel_x];
                    float cur_kernel_val = kernel[kernel_row*kernel_cols+kernel_col];
                    sum += cur_pixel_val*cur_kernel_val;
                }
            }
        }
        output[tid_y*input_cols+tid_x] = sum;
    }
    
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) 
{
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((input_cols+threadsPerBlock.x-1)/threadsPerBlock.x,(input_rows+threadsPerBlock.y-1)/threadsPerBlock.y);
    gaussian_blur_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols);
    cudaDeviceSynchronize();
}
