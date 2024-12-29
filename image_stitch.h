#ifndef IMAGE_STITCH_H
#define IMAGE_STITCH_H

#include <iostream>
#include <fstream>
#include <string>
#include<opencv2/opencv.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include "stitch_gpu.h"
class ImageStitch
{
public:

	ImageStitch();

	~ImageStitch();

	std::vector<cv::detail::CameraParams> findMatrix(std::vector<cv::Mat> images);

	void iniStitch(cv::Mat&firstimg1, cv::Mat&firstimg2, cv::Mat&firstimg3);

	cv::Mat imageStitch(cv::Mat&img1, cv::Mat&img2, cv::Mat&img3);

	int num_images;

};
#endif // STITCH_OPENCL_H
