#include <cuda_runtime.h>

constexpr int threadsPerBlock = 256;

//辅助函数交换
__device__ inline void swap(float& a, float& b) { float tmp=a; a=b; b=tmp; }
//辅助函数对2个数进行升序排列
__device__ inline void sort2(float& a, float& b) { if(b>a) swap(a,b); } 

__global__ void bitonic_block_kernel(float* data, int N) {
    __shared__ float s[2*threadsPerBlock];
    
    int tid = threadIdx.x;
    //当前block内tid线程需要搬运的元素的全局索引，因为一个block处理2*threadsPerblock的数据，因此需要跳过前面处理好的数据
    int idx = blockIdx.x*2*threadsPerBlock + tid;

    //一个线程搬两个数据，中间间隔threadsPerblock
    s[tid] = idx < N ? data[idx] : -1e38f;
    s[tid+threadsPerBlock] = idx+threadsPerBlock < N ? data[idx+threadsPerBlock] : -1e38f;
    __syncthreads();

    //这里的t是跨度
    for(int t=1; t<=threadsPerBlock; t*=2){
        int items=2*t;
        int id=tid/t, pos=tid%t;
        sort2(s[items*id+pos], s[items*id+items-1-pos]);
        __syncthreads();
        for(int r=t/2; r>0; r/=2){
            items=2*r;
            id=tid/r; pos=tid%r;
            sort2(s[items*id+pos], s[items*id+r+pos]);
            __syncthreads();
        }
    }
    if(idx < N) data[idx] = s[tid];
    if(idx+threadsPerBlock < N) data[idx+threadsPerBlock] = s[tid+threadsPerBlock];
}

__global__ void merge_triangle(float* data,int N,int tChunk){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int items=2*tChunk, id=tid/tChunk, pos=tid%tChunk;
    int a=items*id+pos, b=items*id+items-1-pos;
    if(b<N) sort2(data[a],data[b]);
}
__global__ void merge_rhombus(float* data,int N,int tRhombus){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int items=2*tRhombus, id=tid/tRhombus, pos=tid%tRhombus;
    int a=items*id+pos, b=items*id+tRhombus+pos;
    if(b<N) sort2(data[a],data[b]);
}

extern "C" {
void solve(const float* input, float* output, int N, int k) {
    float* d;
    cudaMalloc(&d,N*sizeof(float));
    cudaMemcpy(d,input,N*sizeof(float),cudaMemcpyDeviceToDevice);

    int itemsPerBlock=2*threadsPerBlock;
    int blocks=(N+itemsPerBlock-1)/itemsPerBlock;

    bitonic_block_kernel<<<blocks,threadsPerBlock>>>(d,N);

    for(int tChunk=2*threadsPerBlock;tChunk<N;tChunk*=2){
        int num=tChunk*((N+tChunk-1)/tChunk);
        int b=(num+threadsPerBlock-1)/threadsPerBlock;
        merge_triangle<<<b,threadsPerBlock>>>(d,N,tChunk);
        for(int r=tChunk/2;r>0;r/=2)
            merge_rhombus<<<b,threadsPerBlock>>>(d,N,r);
    }

    cudaDeviceSynchronize();

    int copy_k = (k>N)?N:k;
    cudaMemcpy(output,d,copy_k*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaFree(d);
}
}