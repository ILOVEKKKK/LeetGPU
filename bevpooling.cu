__global__ void bev_pooling(const float* image_feature,const float* depth_weight,float*output,const int* ranks_feature,const int* interval_starts,const int* interval_lens)
{
    

    const int cur_interval = blockIdx.x;//当前block负责的一个BEV网格
    const int cur_channel = threadIdx.x;//当前thread负责的一个特征通道
    const int feature_channel = blockDim.x;

    extern __shared__ smem1[];
    smem1[tid] = image_feature[global_tid]

    float psum = 0.0f;
    const int interval_start = interval_starts[cur_interval];
    const int interval_len = interval_lens[cur_interval];
    #paragma unroll
    for(int i = 0;i<interval_len;++i)
    {
        const int d_weight = depth_weight[interval_start+i];
        const int i_feature = image_feature[in]
        psum = fmaf(d_weight,i_feature,psum);
    }

    const int output_index = 
    output[idx] = 
}