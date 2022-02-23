#include "feature_pyramid.h"

FIMG::feature_pyramid::feature_pyramid()
{

}

FIMG::feature_pyramid::~feature_pyramid()
{

}

int FIMG::feature_pyramid::SetParam(const Fusion_Image_Param& param)
{
	param_ = param;

	return true;
}

int FIMG::feature_pyramid::Gauss_Pyr(const cv::Mat& input, Pyr_FEAT& Img_pyr)
{

	cv::Mat src = input.clone();
	Img_pyr.push_back(src);
	cv::Mat dst;
	for (int i = 0; i < param_.iPyramid_level; i++)
	{
		pyrDown(src, dst, cv::Size(src.cols / 2, src.rows / 2));
		Img_pyr.push_back(dst);
		src = dst;
	}
	return true;
}

int FIMG::feature_pyramid::Laplace_Pyr(Pyr_FEAT& Img_Gaussian_pyr, Pyr_FEAT& Img_Laplacian_pyr)
{
	cv::Mat img_sub, img_up, up, img_lp;

	for (int i = 0; i < param_.iPyramid_level; i++)
	{
		img_sub = Img_Gaussian_pyr[i];
		img_up = Img_Gaussian_pyr[i + 1];
		pyrUp(img_up, up, cv::Size(img_up.cols * 2, img_up.rows * 2));//¶¥²ãÉÏ²ÉÑù 
		img_lp = img_sub - up;
		Img_Laplacian_pyr.push_back(img_lp);
	}

	return true;
}
