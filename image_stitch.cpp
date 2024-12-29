#include "image_stitch.h"

ImageStitch::ImageStitch()
{
	num_images = 3;

}

ImageStitch::~ImageStitch()
{
	releaseMemory();

}

//估计相机参数
std::vector<cv::detail::CameraParams> ImageStitch::findMatrix(std::vector<cv::Mat> images)
{

	std::vector<cv::detail::ImageFeatures> features(num_images);
	cv::Ptr<cv::SIFT> featurefinder = cv::SIFT::create();
	//cv::Ptr<cv::ORB> featurefinder = cv::ORB::create();
	for (int i = 0; i < num_images; ++i)
	{
		computeImageFeatures(featurefinder, images[i], features[i]);
		std::cout << "image #" << i + 1 << "特征点为: " << features[i].keypoints.size() << " 个" << std::endl;
	}

	//特征点匹配
	std::vector<cv::detail::MatchesInfo> pairwise_matches;
	cv::Ptr<cv::detail::FeaturesMatcher> matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(false, 0.3f, 10, 10);
	(*matcher)(features, pairwise_matches);

	//预估相机参数
	cv::Ptr<cv::detail::Estimator> estimator;
	estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();
	std::vector<cv::detail::CameraParams> cameras;
	(*estimator)(features, pairwise_matches, cameras);

	for (int i = 0; i < num_images; ++i)
	{
		cv::Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	//精确相机参数
	cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
	adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
	(*adjuster)(features, pairwise_matches, cameras);

	//波形矫正
	std::vector<cv::Mat> mat;
	for (int i = 0; i < num_images; ++i)
		mat.push_back(cameras[i].R);
	cv::detail::waveCorrect(mat, cv::detail::WAVE_CORRECT_HORIZ); //水平校正
	for (int i = 0; i < num_images; ++i)
		cameras[i].R = mat[i];

	return cameras;

}


//初始化
void ImageStitch::iniStitch(cv::Mat&firstimg1, cv::Mat&firstimg2, cv::Mat&firstimg3)
{
	int srchigh = firstimg1.rows;
	int srcwide = firstimg1.cols;
	std::vector<cv::Mat> firstImages(num_images);
	firstImages[0] = firstimg1;
	firstImages[1] = firstimg2;
	firstImages[2] = firstimg3;
	std::vector<cv::detail::CameraParams> cameras(num_images);
	cameras = findMatrix(firstImages);
	std::vector<cv::Mat> masks(num_images);
	std::vector<cv::Mat> masks_warp(num_images);  //mask扭曲
	std::vector<cv::Mat> images_warp(num_images); //图像扭曲
	std::vector<cv::Point> corners(num_images);   //图像左角点
	std::vector<cv::Size> sizes(num_images);		 //图像尺寸
	cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::CylindricalWarper>();  //柱面投影
	cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(cameras[0].focal);  //因为图像焦距都一样
	std::vector<cv::Mat> x_maps(num_images);
	std::vector<cv::Mat> y_maps(num_images);
	//创建掩码
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(firstImages[i].size(), CV_8U);
		masks[i].setTo(cv::Scalar::all(255));
	}
	//生成映射表、图像位置关系
	for (int i = 0; i < num_images; ++i)
	{
		cv::Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		corners[i] = warper->warp(firstImages[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warp[i]);  //扭曲图像images->images_warp
		warper->buildMaps(firstImages[i].size(), K, cameras[i].R, x_maps[i], y_maps[i]);//生成映射表
		sizes[i] = images_warp[i].size();
		warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warp[i]);  //扭曲masks->masks_warped
	}
	//生成融合权重
	cv::Rect dst_roi_ = cv::detail::resultRoi(corners, sizes);

	cv::Mat dst_weight_map_;
	dst_weight_map_.create(dst_roi_.size(), CV_32F);
	dst_weight_map_.setTo(0);

	std::vector<cv::Mat> weight(num_images);
	std::vector<cv::Mat> weight_map_(num_images);
	float sharpness = 0.01;
	for (int i = 0; i < num_images; ++i)
	{
		cv::detail::createWeightMap(masks_warp[i], sharpness, weight[i]);
		weight[i].copyTo(weight_map_[i]);
	}

	prepareOnGPU(firstimg1, firstimg2, firstimg3, srchigh, srcwide, sizes, x_maps, y_maps, weight_map_, corners, dst_roi_);
}

//进行拼接
cv::Mat ImageStitch::imageStitch(cv::Mat&img1, cv::Mat&img2, cv::Mat&img3)
{
	cv::Mat panorama = workOnGPU(img1, img2, img3);
	return panorama;
}