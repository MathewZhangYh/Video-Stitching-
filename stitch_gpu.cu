#include "stitch_gpu.h"

int srchigh, srcwide;
int maphigh1, mapwide1, maphigh2, mapwide2, maphigh3, mapwide3;
int dsthigh, dstwide;
uchar *dev_srcimg1, *dev_srcimg2, *dev_srcimg3;

uchar *dev_img_map1, *dev_img_map2, *dev_img_map3;
uchar *dev_finalImage;
ushort *dev_dstimg;
int *dev_srcwide, *dev_srchigh, *dev_dstwide, *dev_dsthigh;
int *dev_maphigh1, *dev_mapwide1, *dev_maphigh2, *dev_mapwide2, *dev_maphigh3, *dev_mapwide3;

int *dev_dx1, *dev_dy1, *dev_dx2, *dev_dy2, *dev_dx3, *dev_dy3;
float *dev_x_maps1, *dev_y_maps1, *dev_x_maps2, *dev_y_maps2, *dev_x_maps3, *dev_y_maps3;
float *dev_weight_map1, *dev_weight_map2, *dev_weight_map3, *dev_dst_weight_map;


cv::Mat img_map1, img_map2, img_map3;
cv::Mat dstImage, dst_weight_map, finalImage;

dim3 blockSize;// 设置块大小
dim3 gridSize;// 计算网格大小
dim3 gridSize_dst;// 计算网格大小

__global__ void mapping(uchar * srcImage, uchar * dstImage, float * xmaps, float * ymaps, int  *w, int  *h, int *m_w, int *m_h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dst_w = *w;
	int dst_h = *h;
	int src_w = *m_w;
	int src_h = *m_h;
	if (x >= dst_w || y >= dst_h)
		return;
	int index = y * dst_w + x;
	// 根据映射关系获取输入图像中对应点的像素值
	int inputX = (int)xmaps[index];
	int inputY = (int)ymaps[index];
	int dstindex = index * 3;
	int srcindex = (inputY * src_w + inputX) * 3;
	// 检查输入图像中的坐标是否在有效范围内
	if (inputX >= 0 && inputX < src_w && inputY >= 0 && inputY < src_h)
	{
		// 插值
		dstImage[dstindex] = srcImage[srcindex];
		dstImage[dstindex + 1] = srcImage[srcindex + 1];
		dstImage[dstindex + 2] = srcImage[srcindex + 2];
	}

}
__global__ void feed(uchar * srcImage, ushort * dstImage, float * weight_map, float * dst_weight_map, int  *w, int  *h, int  *ddx, int  *ddy, int *m_w)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dst_w = *w;
	int dst_h = *h;
	int dx = *ddx;
	int dy = *ddy;
	int src_w = *m_w;
	if (x >= dst_w || y >= dst_h)
		return;
	int index = y * dst_w + x;
	float weight = weight_map[index];
	int srcindex = index * 3;
	int index2 = (y + dy) * src_w + (x + dx);
	int dstindex = index2 * 3;//每张图像在全景图上对应的位置不同
	dstImage[dstindex] += static_cast<ushort>(srcImage[srcindex] * weight);
	dstImage[dstindex + 1] += static_cast<ushort>(srcImage[srcindex + 1] * weight);
	dstImage[dstindex + 2] += static_cast<ushort>(srcImage[srcindex + 2] * weight);
	dst_weight_map[index2] += weight;

}
__global__ void blend(ushort * srcImage, uchar * dstImage, float * dst_weight_map, int  *w, int  *h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dst_w = *w;
	int dst_h = *h;
	if (x >= dst_w || y >= dst_h)
		return;
	int index = y * dst_w + x;
	float weight = dst_weight_map[index];
	int dstindex = index * 3;
	dstImage[dstindex] = static_cast<uchar>(srcImage[dstindex] / (weight + 1e-5f));
	dstImage[dstindex + 1] = static_cast<uchar>(srcImage[dstindex + 1] / (weight + 1e-5f));
	dstImage[dstindex + 2] = static_cast<uchar>(srcImage[dstindex + 2] / (weight + 1e-5f));

}


void prepareOnGPU(cv::Mat&srcImage1, cv::Mat&srcImage2, cv::Mat&srcImage3, int m_H, int m_W, std::vector<cv::Size> sizes, std::vector<cv::Mat> x_maps, std::vector<cv::Mat> y_maps,
	std::vector<cv::Mat> weight_map, std::vector<cv::Point> corners, cv::Rect dst_roi)
{

	srchigh = m_H;
	srcwide = m_W;
	maphigh1 = sizes[0].height;
	mapwide1 = sizes[0].width;
	maphigh2 = sizes[1].height;
	mapwide2 = sizes[1].width;
	maphigh3 = sizes[2].height;
	mapwide3 = sizes[2].width;
	dsthigh = dst_roi.size().height;
	dstwide = dst_roi.size().width;


	img_map1 = cv::Mat::zeros(maphigh1, mapwide1, CV_8UC3);
	img_map2 = cv::Mat::zeros(maphigh2, mapwide2, CV_8UC3);
	img_map3 = cv::Mat::zeros(maphigh3, mapwide3, CV_8UC3);
	dstImage = cv::Mat::zeros(dst_roi.size(), CV_16UC3);
	dst_weight_map = cv::Mat::zeros(dst_roi.size(), CV_32FC1);
	finalImage = cv::Mat::zeros(dst_roi.size(), CV_8UC3);

	int dx1 = corners[0].x - dst_roi.x;
	int dy1 = corners[0].y - dst_roi.y;
	int dx2 = corners[1].x - dst_roi.x;
	int dy2 = corners[1].y - dst_roi.y;
	int dx3 = corners[2].x - dst_roi.x;
	int dy3 = corners[2].y - dst_roi.y;
	cudaMalloc((void**)&dev_srcimg1, 3 * srchigh * srcwide * sizeof(uchar));
	cudaMalloc((void**)&dev_srcimg2, 3 * srchigh * srcwide * sizeof(uchar));
	cudaMalloc((void**)&dev_srcimg3, 3 * srchigh * srcwide * sizeof(uchar));


	cudaMalloc((void**)&dev_img_map1, 3 * maphigh1 * mapwide1 * sizeof(uchar));
	cudaMalloc((void**)&dev_img_map2, 3 * maphigh2 * mapwide2 * sizeof(uchar));
	cudaMalloc((void**)&dev_img_map3, 3 * maphigh3 * mapwide3 * sizeof(uchar));

	cudaMalloc((void**)&dev_dstimg, 3 * dsthigh * dstwide * sizeof(ushort));
	cudaMalloc((void**)&dev_finalImage, 3 * dsthigh * dstwide * sizeof(uchar));

	cudaMalloc((void**)&dev_srcwide, sizeof(int));
	cudaMalloc((void**)&dev_srchigh, sizeof(int));
	cudaMalloc((void**)&dev_maphigh1, sizeof(int));
	cudaMalloc((void**)&dev_mapwide1, sizeof(int));
	cudaMalloc((void**)&dev_maphigh2, sizeof(int));
	cudaMalloc((void**)&dev_mapwide2, sizeof(int));
	cudaMalloc((void**)&dev_maphigh3, sizeof(int));
	cudaMalloc((void**)&dev_mapwide3, sizeof(int));

	cudaMalloc((void**)&dev_dsthigh, sizeof(int));
	cudaMalloc((void**)&dev_dstwide, sizeof(int));


	cudaMalloc((void**)&dev_dx1, sizeof(int));
	cudaMalloc((void**)&dev_dy1, sizeof(int));
	cudaMalloc((void**)&dev_dx2, sizeof(int));
	cudaMalloc((void**)&dev_dy2, sizeof(int));
	cudaMalloc((void**)&dev_dx3, sizeof(int));
	cudaMalloc((void**)&dev_dy3, sizeof(int));

	cudaMalloc((void**)&dev_x_maps1, maphigh1 * mapwide1 * sizeof(float));
	cudaMalloc((void**)&dev_y_maps1, maphigh1 * mapwide1 * sizeof(float));
	cudaMalloc((void**)&dev_x_maps2, maphigh2 * mapwide2 * sizeof(float));
	cudaMalloc((void**)&dev_y_maps2, maphigh2 * mapwide2 * sizeof(float));
	cudaMalloc((void**)&dev_x_maps3, maphigh3 * mapwide3 * sizeof(float));
	cudaMalloc((void**)&dev_y_maps3, maphigh3 * mapwide3 * sizeof(float));

	cudaMalloc((void**)&dev_weight_map1, maphigh1 * mapwide1 * sizeof(float));
	cudaMalloc((void**)&dev_weight_map2, maphigh2 * mapwide2 * sizeof(float));
	cudaMalloc((void**)&dev_weight_map3, maphigh3 * mapwide3 * sizeof(float));
	cudaMalloc((void**)&dev_dst_weight_map, dsthigh * dstwide * sizeof(float));

	// 将数据从主机内存复制到GPU内存
	cudaMemcpy(dev_srcimg1, srcImage1.data, 3 * srchigh * srcwide * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srcimg2, srcImage2.data, 3 * srchigh * srcwide * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srcimg3, srcImage3.data, 3 * srchigh * srcwide * sizeof(uchar), cudaMemcpyHostToDevice);



	cudaMemcpy(dev_img_map1, img_map1.data, 3 * maphigh1 * mapwide1 * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_img_map2, img_map2.data, 3 * maphigh2 * mapwide2 * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_img_map3, img_map3.data, 3 * maphigh3 * mapwide3 * sizeof(uchar), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_dstimg, NULL, 3 * dsthigh * dstwide * sizeof(ushort), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_finalImage, NULL, 3 * dsthigh * dstwide * sizeof(uchar), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_srcwide, &srcwide, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srchigh, &srchigh, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_maphigh1, &maphigh1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mapwide1, &mapwide1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_maphigh2, &maphigh2, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mapwide2, &mapwide2, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_maphigh3, &maphigh3, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mapwide3, &mapwide3, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_dsthigh, &dsthigh, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dstwide, &dstwide, sizeof(int), cudaMemcpyHostToDevice);


	cudaMemcpy(dev_dx1, &dx1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dy1, &dy1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dx2, &dx2, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dy2, &dy2, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dx3, &dx3, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dy3, &dy3, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_x_maps1, reinterpret_cast<float*>(x_maps[0].data), maphigh1 * mapwide1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_maps1, reinterpret_cast<float*>(y_maps[0].data), maphigh1 * mapwide1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x_maps2, reinterpret_cast<float*>(x_maps[1].data), maphigh2 * mapwide2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_maps2, reinterpret_cast<float*>(y_maps[1].data), maphigh2 * mapwide2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x_maps3, reinterpret_cast<float*>(x_maps[2].data), maphigh3 * mapwide3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_maps3, reinterpret_cast<float*>(y_maps[2].data), maphigh3 * mapwide3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_weight_map1, reinterpret_cast<float*>(weight_map[0].data), maphigh1 * mapwide1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weight_map2, reinterpret_cast<float*>(weight_map[1].data), maphigh2 * mapwide2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weight_map3, reinterpret_cast<float*>(weight_map[2].data), maphigh3 * mapwide3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dst_weight_map, NULL, dsthigh * dstwide * sizeof(float), cudaMemcpyHostToDevice);


	// 设置块大小
	//dim3 blockSize(32, 16, 1);
	blockSize.x = 32;
	blockSize.y = 24;
	blockSize.z = 1;
	// 计算网格大小
	int gridX = (srcwide + blockSize.x - 1) / blockSize.x;
	int gridY = (srchigh + blockSize.y - 1) / blockSize.y;
	//dim3 gridSize(gridX, gridY, 1);
	gridSize.x = gridX;
	gridSize.y = gridY;
	gridSize.z = 1;
	// 计算网格大小
	int gridX_dst = (dstwide + blockSize.x - 1) / blockSize.x;
	int gridY_dst = (dsthigh + blockSize.y - 1) / blockSize.y;
	//dim3 gridSize(gridX, gridY, 1);
	gridSize_dst.x = gridX_dst;
	gridSize_dst.y = gridY_dst;
	gridSize_dst.z = 1;
}

cv::Mat workOnGPU(cv::Mat&srcImage1, cv::Mat&srcImage2, cv::Mat&srcImage3)
{

	cudaMemcpy(dev_srcimg1, srcImage1.data, 3 * srchigh * srcwide * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srcimg2, srcImage2.data, 3 * srchigh * srcwide * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_srcimg3, srcImage3.data, 3 * srchigh * srcwide * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemset(dev_dst_weight_map, 0, dsthigh * dstwide * sizeof(float));
	cudaMemset(dev_dstimg, 0, 3 * dsthigh * dstwide * sizeof(ushort));

	mapping << < gridSize, blockSize >> > (dev_srcimg1, dev_img_map1, dev_x_maps1, dev_y_maps1, dev_mapwide1, dev_maphigh1, dev_srcwide, dev_srchigh);
	mapping << < gridSize, blockSize >> > (dev_srcimg2, dev_img_map2, dev_x_maps2, dev_y_maps2, dev_mapwide2, dev_maphigh2, dev_srcwide, dev_srchigh);
	mapping << < gridSize, blockSize >> > (dev_srcimg3, dev_img_map3, dev_x_maps3, dev_y_maps3, dev_mapwide3, dev_maphigh3, dev_srcwide, dev_srchigh);

	feed << < gridSize, blockSize >> > (dev_img_map1, dev_dstimg, dev_weight_map1, dev_dst_weight_map, dev_mapwide1, dev_maphigh1, dev_dx1, dev_dy1, dev_dstwide);
	feed << < gridSize, blockSize >> > (dev_img_map2, dev_dstimg, dev_weight_map2, dev_dst_weight_map, dev_mapwide2, dev_maphigh2, dev_dx2, dev_dy2, dev_dstwide);
	feed << < gridSize, blockSize >> > (dev_img_map3, dev_dstimg, dev_weight_map3, dev_dst_weight_map, dev_mapwide3, dev_maphigh3, dev_dx3, dev_dy3, dev_dstwide);

	blend << < gridSize_dst, blockSize >> > (dev_dstimg, dev_finalImage, dev_dst_weight_map, dev_dstwide, dev_dsthigh);

	cudaMemcpy(finalImage.data, dev_finalImage, 3 * dsthigh * dstwide * sizeof(uchar), cudaMemcpyDeviceToHost);

	return finalImage;
}

void releaseMemory()
{
	// 释放GPU内存
	cudaFree(dev_srcimg1);
	cudaFree(dev_srcimg2);
	cudaFree(dev_srcimg3);

	cudaFree(dev_img_map1);
	cudaFree(dev_img_map2);
	cudaFree(dev_img_map3);
	cudaFree(dev_dstimg);
	cudaFree(dev_finalImage);
	cudaFree(dev_srcwide);
	cudaFree(dev_srchigh);
	cudaFree(dev_dstwide);
	cudaFree(dev_dsthigh);
	cudaFree(dev_maphigh1);
	cudaFree(dev_mapwide1);
	cudaFree(dev_maphigh2);
	cudaFree(dev_mapwide2);
	cudaFree(dev_maphigh3);
	cudaFree(dev_mapwide3);

	cudaFree(dev_dx1);
	cudaFree(dev_dy1);
	cudaFree(dev_dx2);
	cudaFree(dev_dy2);
	cudaFree(dev_dx3);
	cudaFree(dev_dy3);
	cudaFree(dev_x_maps1);
	cudaFree(dev_y_maps1);
	cudaFree(dev_x_maps2);
	cudaFree(dev_y_maps2);
	cudaFree(dev_x_maps3);
	cudaFree(dev_y_maps3);
	cudaFree(dev_weight_map1);
	cudaFree(dev_weight_map2);
	cudaFree(dev_weight_map3);
	cudaFree(dev_dst_weight_map);

}

