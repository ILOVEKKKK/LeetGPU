#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/numeric/numeric_types.hpp>

using namespace cute;

template<typename T,typename BlockShape,typename TiledMma,typename TiledCopy>
__global__ void gemm_cute_kernel(const T* A,const T* B,const T* Bias,T* C,int M,int N,int K)
{
    Tensor mA = make_tensor(make_gmem_ptr(A),make_shape(M,K),LayoutRight{});
    Tensor mB = make_tensor(make_gmem_ptr(B),make_shape(N,K),LayoutRight{});
    Tensor mC = make_tensor(make_gmem_ptr(C),make_shape(M,N),LayoutRight{});
    Tensor mD = make_tensor(make_gmem_ptr(Bias),make_shape(M,N),LayoutRight{});

    //获取当前block的坐标以及block的具体大小
    //block_shape是一个tuple，包含了block在每个维度上的切分大小，TileM+TileN+TileK
    auto block_shape = BlockShape{};
    auto block_coord = make_coord(blockIdx.x,blockIdx.y);

    //参数1:：要切分的Tensor  参数2：切分的大小  参数3：当前block负责的子矩阵
    Tensor gA = local_tile(mA,select<0,2>(block_shape),make_coord(blockIdx.x,_));
    Tensor gB = local_tile(mB,select<1,2>(block_shape),make_coord(blockIdx.y,_));
    Tensor gC = local_tile(mC,select<0,1>(block_shape),make_coord(blockIdx.x,blockIdx.y));
    Tensor gD = local_tile(mD,select<0,1>(block_shape),make_coord(blockIdx.x,blockIdx.y));

    __shared__ T smemA[size<0>(block_shape)*size<2>(block_shape)];
    __shared__ T smemB[size<1>(block_shape)*size<2>(block_shape)];
    Tensor sA = make_tensor(make_smem_ptr(smemA),make_shape(size<0>(block_shape),size<2>(block_shape)),LayoutRight{});
    Tensor sB = make_tensor(make_smem_ptr(smemB),make_shape(size<1>(block_shape),size<2>(block_shape)),LayoutRight{});

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCgD = thr_mma.partition_C(gD);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);
    
    clear(tCrC);
    copy(tCgD,tCrC);

    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy.partition_S(gA);
    Tensor tBgB = thr_copy.partition_S(gB);
    Tensor tAsA = thr_copy.partition_D(sA);
    Tensor tBsB = thr_copy.partition_D(sB);

    int k_tiles = size<2>(gA);
    for(int i = 0;i < k_tiles;++i)
    {
        copy(tiled_copy,tAgA(_,_,_,i),tAsA);
        copy(tiled_copy,tBgB(_,_,_,i),tBsB);
        __syncthreads();

        copy(tCsA,tCrA);
        copy(tCsB,tCrB);

        gemm(tiled_mma,tCrA,tCrB,tCrC);
        __syncthreads();
    }
    copy(tCrC,tCgC);
}

void run_gemm()
{
    int M = 1024;
    int N = 1024;
    int K = 2048;

    using BLK_M = Int<128>;
    using BLK_N = Int<128>;
    using BLK_K = Int<32>;
    using ThreadBlockShape = Shape<BLK_M,BLK_N,BLK_K>;

    // 1. 使用 make_tiled_mma 来自动推导 TiledMma 类型
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>{},
        Layout<Shape<_4, _1, _1>>{},  // MMA Atom Layout
        Tile<_128, _128, _32>{}        // Threadblock Tile
    );
    using TiledMma = decltype(tiled_mma);

    // 2. 使用 make_tiled_copy 来自动推导 TiledCopy 类型 (解决你的报错)
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<DefaultCopy, float>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thread Layout: 128 threads
        Layout<Shape<_1, _1>>{}                   // Value Layout (通常设为 1x1)
    );
    using TiledCopy = decltype(tiled_copy);

    thrust::device_vector<float> dA(M * K, 1.0f);
    thrust::device_vector<float> dB(N * K, 1.0f);
    thrust::device_vector<float> dC(M * N, 0.0f);
    thrust::device_vector<float> dBias(M * N, 1000.0f);

    dim3 grid(M / 128, N / 128);
    dim3 block(128); 

    gemm_cute_kernel<float, ThreadBlockShape, TiledMma, TiledCopy>
        <<<grid, block>>>(
            thrust::raw_pointer_cast(dA.data()),
            thrust::raw_pointer_cast(dB.data()),
            thrust::raw_pointer_cast(dBias.data()),
            thrust::raw_pointer_cast(dC.data()),
            M, N, K
        );

    cudaDeviceSynchronize();
    thrust::host_vector<float> hC = dC;
    auto layout_C = make_layout(make_shape(M, N), LayoutRight{});
    auto tensor_C = make_tensor(hC.data(), layout_C);
    print_tensor(tensor_C);
}

int main()
{
    run_gemm();
    return 0;
}