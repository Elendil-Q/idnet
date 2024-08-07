#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include "cuda_utils.hpp"
#include "random"

using namespace std;

namespace py = pybind11;

void free_memory()
{
    free_gpu_memory_mc();
}

Eigen::MatrixXf motion_compensation_gpu(vector<float> evt_x, vector<float> evt_y, vector<uint8_t> evt_p, vector<double> evt_t, Eigen::MatrixXf optic_x, Eigen::MatrixXf optic_y, const int height, const int width)
{

    // x的长度为事件的数量
    int events_num = evt_x.size();

    const int threads_per_block = 64;
    const int blocks_per_grid = (events_num + threads_per_block - 1) / threads_per_block;

    double min_t = *std::min_element(evt_t.begin(), evt_t.end());
    double max_t = *std::max_element(evt_t.begin(), evt_t.end());
    // normalize t
    for (int i = 0; i < events_num; i++)
    {
        evt_t[i] = (evt_t[i] - min_t) / (max_t - min_t);
    }

    Eigen::MatrixXf iwe_mat = iwe_cuda_warp(evt_x, evt_y, evt_p, evt_t, optic_x, optic_y, width, height);

    return iwe_mat;
}

PYBIND11_MODULE(warp_event, m)
{
    m.doc() = "pybind11 cuda warp plugin"; // optional module docstring
    m.def("motion_compensation_gpu", &motion_compensation_gpu, "motion compensation using gpu");
    m.def("free_memory", &free_memory, "free memory");
}