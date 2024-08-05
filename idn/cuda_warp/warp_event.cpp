#include <opencv2/core/core.hpp>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include "cuda_utils.hpp"
using namespace std;
using namespace cv;

namespace py = pybind11;

void free_memory()
{
    free_gpu_memory_mc();
}

py::array_t<float> motion_compensation_gpu(py::array_t<float> x, py::array_t<float> y, py::array_t<uint8_t> p, py::array_t<double> t, py::array_t<float> optic_flow_py, const int height, const int width)
{

    py::buffer_info buf_info_x = x.request();
    float *ptr = static_cast<float *>(buf_info_x.ptr);
    auto shape = buf_info_x.shape;
    int events_num = shape[0];

    const int threads_per_block = 64;
    const int blocks_per_grid = (events_num + threads_per_block - 1) / threads_per_block;

    int length = buf_info_x.size;
    vector<float> evt_x(ptr, ptr + length);

    // 将pybind11的event数据转换为vector的event数据
    py::buffer_info buf_info_y = y.request();
    float *ptr_y = static_cast<float *>(buf_info_y.ptr);
    length = buf_info_y.size;
    vector<float> evt_y(ptr_y, ptr_y + length);

    py::buffer_info buf_info_p = p.request();
    uint8_t *ptr_p = static_cast<uint8_t *>(buf_info_p.ptr);
    length = buf_info_p.size;
    vector<uint8_t> evt_p(ptr_p, ptr_p + length);

    py::buffer_info buf_info_t = t.request();
    double *ptr_t = static_cast<double *>(buf_info_t.ptr);
    length = buf_info_t.size;
    vector<double> evt_t(ptr_t, ptr_t + length);

    double min_t = *std::min_element(evt_t.begin(), evt_t.end());
    double max_t = *std::max_element(evt_t.begin(), evt_t.end());
    // normalize t
    for (int i = 0; i < events_num; i++)
    {
        evt_t[i] = (evt_t[i] - min_t) / (max_t - min_t);
    }

    // 将pybind11的optic flow数据转换为Mat的optic flow数据
    py::buffer_info buf_info_optic = optic_flow_py.request();
    float *ptr_optic = static_cast<float *>(buf_info_optic.ptr);
    auto shape_optic = buf_info_optic.shape;
    int optic_height = shape_optic[0];
    int optic_width = shape_optic[1];
    int optic_channels = shape_optic[2];
    int optic_bytes = optic_height * optic_width * optic_channels * sizeof(float);
    Mat optic_flow(optic_height, optic_width, CV_32FC2, ptr_optic);

    Mat iwe_mat = iwe_cuda_warp(evt_x, evt_y, evt_p, evt_t, optic_flow, width, height);

    return py::array_t<float>({iwe_mat.rows, iwe_mat.cols}, iwe_mat.ptr<float>());
}

PYBIND11_MODULE(warp_events, m)
{
    m.doc() = "pybind11 cuda warp plugin"; // optional module docstring
    m.def("motion_compensation_gpu", &motion_compensation_gpu, "motion compensation using gpu");
    m.def("free_memory", &free_memory, "free memory");
}