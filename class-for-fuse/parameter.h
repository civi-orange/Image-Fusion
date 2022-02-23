#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <xfeatures2d/nonfree.hpp>

#include <direct.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include "time.h"
#include <stdio.h>

using std::vector;
using std::string;

namespace FIMG
{

typedef std::vector<cv::Mat> Pyr_FEAT;

struct fusion_image_index //融合评价指标
{
	double fusion_EN = 0.0;				//熵
	double fusion_MI = 0.0;				//融合互信息
	double fusion_meanGrad = 0.0;		//平均梯度
	double fusion_SSIM_irf = 0.0;		//结构相似度
	double fusion_SSIM_visf = 0.0;		//
	double fusion_SF = 0.0;				//空间频率
	double fusion_SD = 0.0;				//对比度
	double fusion_CC_irf = 0.0;			//相关系数
	double fusion_CC_visf = 0.0;
	double fusion_VIF = 0.0;			//视觉信息保真度
	double fusion_Qab_f = 0.0;			//边缘信息保存量
};

struct affine_trans_params
{
	double x_scale = 1.0;		//x 轴缩放倍率
	double y_scale = 1.0;		//y 轴缩放倍数
	double rotate_angle = 0.0;	//旋转角度，degree；正值为逆时针旋转
	double x_offset = 0.0;		//x 轴平移量
	double y_offset = 0.0;		//y 轴平移量

};

struct Fusion_Image_Param
{
	int iPyramid_level = 4;
	double dAlpha = 0.7;
	double dGamma = 1.8;

	int iThresh_max_EN = 0;//最大熵分割初始阈值
	int iMax_EN_p = 10;//最大熵分割补偿阈值
	bool init_flag = false;

	affine_trans_params affine_param;
	fusion_image_index  image_indictor;

};


}