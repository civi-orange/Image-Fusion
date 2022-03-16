#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <xfeatures2d/nonfree.hpp>

#include "opencv2/ximgproc.hpp" //引导滤波

#include <direct.h>
#include <io.h>
#include <fstream>
#include <iostream>
#include "time.h"
#include <stdio.h>

using std::vector;
using std::string;

#pragma warning(disable:4996) //获取时间函数失效警告去除

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

	//相机标定单应性矩阵
	cv::Mat homo_mat_LFuse = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 1, 1, 0, 0, 1); //左微光+红外	
	cv::Mat homo_mat_RFuse = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 1, 1, 0, 0, 1); //右微光+红外	
	cv::Mat homo_mat_LRVision = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 1, 1, 0, 0, 1); //左右微光立体视觉参数矩阵

};


}

namespace SMAG //Stereo matching algorithm group
{

	struct BM_param
	{
		int preFiletertype = cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE;// PREFILTER_NORMALIZED_RESPONSE:0 or PREFILTER_XSOBEL:1
		int preFilterSize = 9;
		int preFileterCap = 31;
		int textureThreshold = 10;
		int UniquenessRatio = 7;
		int speckleWindowSize = 20;
		int smallblockSize = 3;
		cv::Rect roi_left, roi_right;


		int blockSize = 5; //必须为＞1的奇数 一般取 5-21
		int minDisparity = 0; //最小视差
		int numDisparities_ratio = 10;// numDisparities = 16 * ratio 视差范围*倍率
		int SpeckleRange = 32;//
		int disp12MaxDiff = 1;//
		
		BM_param clone()
		{
			BM_param temp;

			temp.blockSize = blockSize;
			temp.minDisparity = minDisparity;
			temp.numDisparities_ratio = numDisparities_ratio;
			temp.SpeckleRange = SpeckleRange;
			temp.disp12MaxDiff = disp12MaxDiff;

			temp.preFiletertype = preFiletertype;
			temp.preFilterSize = preFilterSize;
			temp.preFileterCap = preFileterCap;
			temp.textureThreshold = textureThreshold;
			temp.UniquenessRatio = UniquenessRatio;
			temp.speckleWindowSize = speckleWindowSize;
			temp.smallblockSize = smallblockSize;
			temp.roi_left = roi_left;
			temp.roi_right = roi_right;

			return temp;
		}

	};

	struct SGBM_param
	{
		int mode = cv::StereoSGBM::MODE_SGBM;// MODE_SGBM=0, MODE_HH=1, MODE_SGBM_3WAY=2, MODE_HH4=3
		int preFileterCap = 31;
		int UniquenessRatio = 7;
		int P1 = 600; //P1 = 8 * left.channels() * blockSize* blockSize 平滑系数
		int P2 = 2400;


		int blockSize = 5; //必须为＞1的奇数 一般取 5-21
		int minDisparity = 0;
		int numDisparities_ratio = 10;// numDisparities = 16 * ratio
		int SpeckleRange = 32;
		int speckleWindowSize = 100;
		int disp12MaxDiff = 1;

		SGBM_param clone()
		{
			SGBM_param temp;

			temp.blockSize = blockSize;
			temp.minDisparity = minDisparity;
			temp.numDisparities_ratio = numDisparities_ratio;
			temp.SpeckleRange = SpeckleRange;
			temp.speckleWindowSize = speckleWindowSize;
			temp.disp12MaxDiff = disp12MaxDiff;

			temp.mode = mode;
			temp.preFileterCap = preFileterCap;
			temp.P1 = P1;
			temp.P2 = P2;
			temp.UniquenessRatio = UniquenessRatio;

			return temp;
		}

	};

}