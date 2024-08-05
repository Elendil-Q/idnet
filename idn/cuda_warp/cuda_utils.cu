#include "cuda_utils.hpp"

__global__ void warp_events(float *evt_x, float *evt_y, double *t, uint8_t *p, const float *optic_x, const float *optic_y, int cols, int evt_num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < evt_num)
    {
        // motion compensation
        int x = round(evt_x[idx]);
        int y = round(evt_y[idx]);
        double dt = 1.00 - t[idx];
        float x_new = x + optic_x[y * cols + x] * dt;
        float y_new = y + optic_y[y * cols + x] * dt;
        evt_x[idx] = x_new;
        evt_y[idx] = y_new;
    }
}

__global__ void reset_iwe(float *iwe, int cols, int rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols * rows)
    {
        iwe[idx] = 0;
    }
}

__device__ void inbounds(int x, int y, int cols, int rows, bool &in)
{
    in = x >= 0 && x < cols && y >= 0 && y < rows;
}

__global__ void generate_iwe(float *evt_x, float *evt_y, uint8_t *p, int cols, int rows, int evt_num, float *iwe)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < evt_num)
    {
        float x = evt_x[idx];
        float y = evt_y[idx];
        int top_y = y;
        int bot_y = y + 1;
        int left_x = x;
        int right_x = x + 1;
        // 双线性插值，计算各个像素的权重
        float w_tl = (1 - (x - left_x)) * (1 - (y - top_y));
        float w_tr = (x - left_x) * (1 - (y - top_y));
        float w_bl = (1 - (x - left_x)) * (y - top_y);
        float w_br = (x - left_x) * (y - top_y);
        bool in;
        inbounds(left_x, top_y, cols, rows, in);
        if (in)
        {
            iwe[top_y * cols + left_x] += w_tl;
        }
        inbounds(right_x, top_y, cols, rows, in);
        if (in)
        {
            iwe[top_y * cols + right_x] += w_tr;
        }
        inbounds(left_x, bot_y, cols, rows, in);
        if (in)
        {
            iwe[bot_y * cols + left_x] += w_bl;
        }
        inbounds(right_x, bot_y, cols, rows, in);
        if (in)
        {
            iwe[bot_y * cols + right_x] += w_br;
        }
    }
}

void init_gpu_memory_mc(Mat optic_flow)
{
    Mat iwe_mat = cv::Mat::zeros(optic_flow.rows, optic_flow.cols, CV_32FC1);
    Mat optic_flow_channels[2];
    split(optic_flow, optic_flow_channels);
    optic_bytes = optic_flow_channels[0].rows * optic_flow_channels[0].step;
    iwe_bytes = iwe_mat.rows * iwe_mat.step;
    cudaMalloc<float>(&optic_x, optic_bytes);
    cudaMalloc<float>(&optic_y, optic_bytes);
    cudaMalloc<float>(&iwe, iwe_bytes);
}

void free_gpu_memory_mc()
{
    if (memory_initialized_mc)
    {
        cudaFree(optic_x);
        cudaFree(optic_y);
        cudaFree(iwe);
    }
}

/**
 * @brief 利用CUDA加速实现事件流的运动补偿
 *
 * @param evt_x [n] 事件x坐标 float
 * @param evt_y [n] 事件y坐标 float
 * @param evt_p [n] 事件极性 uint8_t
 * @param evt_t [n] 事件时间 double
 * @param optic_flow [H,W,2] 光流图像 CV_32FC2
 * @param width 图像宽度
 * @param height 图像高度
 * @return Mat [H,W] 运动补偿后的事件图像 CV_32FC1
 */
Mat iwe_cuda_warp(vector<float> evt_x, vector<float> evt_y, vector<uint8_t> evt_p, vector<double> evt_t, Mat optic_flow, const int width, const int height)
{
    int events_num = evt_x.size();
    const int threads_per_block = 64;
    const int blocks_per_grid = (events_num + threads_per_block - 1) / threads_per_block;
    // 初始化GPU内存
    if (!memory_initialized_mc)
    {
        init_gpu_memory_mc(optic_flow);
        memory_initialized_mc = true;
    }

    double *evt_t_d;
    cudaMalloc<double>(&evt_t_d, events_num * sizeof(double));
    cudaMemcpy(evt_t_d, evt_t.data(), events_num * sizeof(double), cudaMemcpyHostToDevice);

    // copy events to GPU
    float *evt_x_d, *evt_y_d;
    uint8_t *evt_p_d;

    cudaMalloc<float>(&evt_x_d, events_num * sizeof(float));
    cudaMalloc<float>(&evt_y_d, events_num * sizeof(float));
    cudaMalloc<uint8_t>(&evt_p_d, events_num * sizeof(uint8_t));
    cudaMemcpy(evt_x_d, evt_x.data(), events_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(evt_y_d, evt_y.data(), events_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(evt_p_d, evt_p.data(), events_num * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // copy optic flow to GPU
    Mat optic_flow_channels[2];
    split(optic_flow, optic_flow_channels);
    cudaMemcpy(optic_x, optic_flow_channels[0].data, optic_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(optic_y, optic_flow_channels[1].data, optic_bytes, cudaMemcpyHostToDevice);

    // reset iwe
    const int threads_per_block_iwe = 64;
    const int blocks_per_grid_iwe = (width * height + threads_per_block - 1) / threads_per_block;
    reset_iwe<<<blocks_per_grid_iwe, threads_per_block_iwe>>>(iwe, width, height);
    cudaDeviceSynchronize();

    // warp
    warp_events<<<blocks_per_grid, threads_per_block>>>(evt_x_d, evt_y_d, evt_t_d, evt_p_d, optic_x, optic_y, width, events_num);
    cudaDeviceSynchronize();

    // generate iwe
    generate_iwe<<<blocks_per_grid, threads_per_block>>>(evt_x_d, evt_y_d, evt_p_d, width, height, events_num, iwe);
    cudaDeviceSynchronize();

    float *evt_y_host = new float[events_num];
    cudaMemcpy(evt_y_host, evt_y_d, events_num * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] evt_y_host;
    // copy iwe to CPU
    Mat iwe_mat = cv::Mat::zeros(height, width, CV_32F);
    cudaMemcpy(iwe_mat.data, iwe, iwe_bytes, cudaMemcpyDeviceToHost);
    

    // free memory
    cudaFree(evt_x_d);
    cudaFree(evt_y_d);
    cudaFree(evt_p_d);
    cudaFree(evt_t_d);
    return iwe_mat;
}