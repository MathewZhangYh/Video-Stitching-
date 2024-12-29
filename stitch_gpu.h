#pragma once
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<opencv2/opencv.hpp>


__global__ void mapping(uchar * srcImage, uchar * dstImage, float * xmaps, float * ymaps, int  *w, int  *h, int *m_w, int *m_h);
__global__ void feed(uchar * srcImage, ushort * dstImage, float * weight_map, float * dst_weight_map, int  *w, int  *h, int  *ddx, int  *ddy, int *m_w);
__global__ void blend(ushort * srcImage, uchar * dstImage, float * dst_weight_map, int  *w, int  *h);

void prepareOnGPU(cv::Mat&srcImage1, cv::Mat&srcImage2, cv::Mat&srcImage3, int m_H, int m_W,
	std::vector<cv::Size> sizes, std::vector<cv::Mat> x_maps, std::vector<cv::Mat> y_maps, std::vector<cv::Mat> weight_map,
	std::vector<cv::Point> corners, cv::Rect dst_roi);

cv::Mat workOnGPU(cv::Mat&srcImage1, cv::Mat&srcImage2, cv::Mat&srcImage3);

void releaseMemory();