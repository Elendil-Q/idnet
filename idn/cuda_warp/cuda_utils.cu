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
        int top_y = (int)floor(y);
        int bot_y = top_y + 1;
        int left_x = (int)floor(x);
        int right_x = left_x + 1;
        // 双线性插值，计算各个像素的权重
        float delta_x = x - (float)left_x;
        float delta_y = y - (float)top_y;
        float w_tl = (1.0 - delta_x) * (1.0 - delta_y);
        float w_tr = delta_x * (1.0 - delta_y);
        float w_bl = (1.0 - delta_x) * delta_y;
        float w_br = delta_x * delta_y;
        bool in = false;
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

void init_gpu_memory_mc(Eigen::MatrixXf optic_flow_x, Eigen::MatrixXf optic_flow_y)
{
    Eigen::MatrixXf optic_flow(optic_flow_x.rows(), optic_flow_x.cols() * 2);
    Eigen::MatrixXf zero_iwe = Eigen::MatrixXf::Zero(optic_flow_x.rows(), optic_flow_x.cols());
    optic_bytes = optic_flow_x.rows() * optic_flow_x.cols() * sizeof(float);
    iwe_bytes = optic_bytes;
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
Eigen::MatrixXf iwe_cuda_warp(vector<float> evt_x, vector<float> evt_y, vector<uint8_t> evt_p, vector<double> evt_t, Eigen::MatrixXf optic_flow_x, Eigen::MatrixXf optic_flow_y, const int width, const int height)
{
    int events_num = evt_x.size();
    const int threads_per_block = 64;
    const int blocks_per_grid = (events_num + threads_per_block - 1) / threads_per_block;
    // 初始化GPU内存
    if (!memory_initialized_mc)
    {
        init_gpu_memory_mc(optic_flow_x, optic_flow_y);
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
    cudaMemcpy(optic_x, optic_flow_x.data(), optic_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(optic_y, optic_flow_y.data(), optic_bytes, cudaMemcpyHostToDevice);

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
    Eigen::MatrixXf iwe_mat = Eigen::MatrixXf::Zero(height, width);

    cudaMemcpy(iwe_mat.data(), iwe, iwe_bytes, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(evt_x_d);
    cudaFree(evt_y_d);
    cudaFree(evt_p_d);
    cudaFree(evt_t_d);
    return iwe_mat;
}