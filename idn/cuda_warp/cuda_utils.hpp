#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

/**
 * global variables definition
 */
bool memory_initialized_mc = false;
float *iwe;
float *optic_x, *optic_y;
int optic_bytes;
int iwe_bytes;

void init_gpu_memory_mc(Mat optic_flow);

void free_gpu_memory_mc();

Mat iwe_cuda_warp(vector<float> x, vector<float> y, vector<uint8_t> p, vector<double> t, Mat flow, const int width, const int height);