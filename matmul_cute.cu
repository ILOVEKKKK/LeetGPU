#include<iostream>
#include<vector>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

#include<cute/tensor.hpp>
#include<cute/numeric/numeric_types.hpp>

using namespace cute;

template<typename T,typename ThreadBlockShape,typename TiledMma, typename TiledCopy>
__global__ void gemm_cute_kernel(T const* A_ptr,T const* B_ptr, T* C_ptr, int M, int N,int K)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A_ptr),make_shape(M,K),make_stride(K,1));
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr),make_shape(N,K),make_stride(K,1));
    Tensor mC = make_tensor(make_gmem_ptr(C_ptr),make_shape(M,N),make_stride(N,1));

    auto block_shape = ThreadBlockShape{};
    auto block_coord = make_coord(blockIdx.x,blockIdx.y);

    Tensor gA = local_tile(mA,select<0,2>(block_shape),make_coord(get<0>(block_coord),_));
    Tensor gB = local_tile(mB,select<1,2>(block_shape),make_coord(get<1>(block_coord),_));
    Tensor gC = local_tile(mC,select<0,1>(block_shape),block_coord);

    __shared__ T smemA[size<0>(block_shape)*size<2>(block_shape)];
    __shared__ T smemB[size<1>(block_shape)*size<2>(block_shape)];
    Tensor sA = make_tensor(make_smem_ptr(smemA),make_layout(select<0,2>(block_shape),LayoutRight{}));
    Tensor sB = make_tensor(make_smem_ptr(smemB),make_layout(select<1,2>(block_shape),LayoutRight{}));

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);

    clear(tCrC);

    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy.partition_S(gA);
    Tensor tAsA = thr_copy.partition_D(sA);
    Tensor tBgB = thr_copy.partition_S(gB);
    Tensor tBsB = thr_copy.partition_D(sB);

    int k_tiles = size<2>(gA);
    for(int k = 0;k < k_tiles;++k)
    {
        copy(tiled_copy, tAgA(_,_,_,k), tAsA);
        copy(tiled_copy, tBgB(_, _, _, k), tBsB);
        __syncthreads();

        copy(tCsA,tCrA);
        copy(tCsB,tCrB);

        gemm(tiled_mma,tCrA,tCrB,tCrC);
        __syncthreads();
    }
    copy(tCrC, tCgC);
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
        Layout<Shape<_2, _2, _1>>{},  // MMA Atom Layout
        Tile<_128, _128, _32>{}        // Threadblock Tile
    );
    using TiledMma = decltype(tiled_mma);

    // 2. 使用 make_tiled_copy 来自动推导 TiledCopy 类型 (解决你的报错)
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<DefaultCopy, float>{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{}, // Thread Layout: 128 threads
        Layout<Shape<_1, _1>>{}                   // Value Layout (通常设为 1x1)
    );
    using TiledCopy = decltype(tiled_copy);
    
        // 分配内存
    thrust::device_vector<float> dA(M * K, 1.0f);
    thrust::device_vector<float> dB(N * K, 1.0f);
    thrust::device_vector<float> dC(M * N, 0.0f);

    dim3 grid(M / 128, N / 128);
    dim3 block(128); // 32*4 threads

    gemm_cute_kernel<float, ThreadBlockShape, TiledMma, TiledCopy>
        <<<grid, block>>>(
            thrust::raw_pointer_cast(dA.data()),
            thrust::raw_pointer_cast(dB.data()),
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