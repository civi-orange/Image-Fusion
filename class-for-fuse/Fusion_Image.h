#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <xfeatures2d/nonfree.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include "time.h"
#include <stdio.h>
#include <io.h>
#include <direct.h>

#include "parameter.h"
#include "feature_pyramid.h"
#include "method_group.h"

namespace FIMG
{

class Fusion_Image
{

public:

	Fusion_Image();

	~Fusion_Image();

	int init_parameter(const FIMG::Fusion_Image_Param &parameter);

	int init_fusion(const cv::Mat& img_std);

	int fusion_image(const cv::Mat& img_front, const cv::Mat& img_back, cv::Mat& img_dst);

	int Get_Fusion_indicator(cv::Mat img_front, cv::Mat img_back, cv::Mat img_fuse, fusion_image_index& fuse_indicator);




private:

	int laplace_pyr_fusion(FIMG::Pyr_FEAT& Img_lp_front, FIMG::Pyr_FEAT& Img_lp_back, FIMG::Pyr_FEAT& mask_gau, FIMG::Pyr_FEAT& blend_lp);



private:
	
	Fusion_Image_Param param_;

	feature_pyramid feature_;

	Pyr_FEAT mask_Pyr_;

	Pyr_FEAT front_Gauss_Pyr_;
	Pyr_FEAT back_Gauss_Pyr_;

	Pyr_FEAT front_Laplace_Pyr_;
	Pyr_FEAT back_Laplace_Pyr_;

	method_group method_;

};

}
