#ifndef CUDA_UTILS_HPP
#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

/**
 * global variables definition
 */
bool memory_initialized_mc = false;
float *iwe;
float *optic_x, *optic_y;
int optic_bytes;
int iwe_bytes;

void init_gpu_memory_mc(Eigen::MatrixXf optic_flow);

void free_gpu_memory_mc();

Eigen::MatrixXf iwe_cuda_warp(vector<float> evt_x, vector<float> evt_y, vector<uint8_t> evt_p, vector<double> evt_t, Eigen::MatrixXf optic_flow_x, Eigen::MatrixXf optic_flow_y, const int width, const int height);

#endif