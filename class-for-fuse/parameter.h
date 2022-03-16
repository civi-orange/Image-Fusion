#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <xfeatures2d/nonfree.hpp>

#include "opencv2/ximgproc.hpp" //�����˲�

#include <direct.h>
#include <io.h>
#include <fstream>
#include <iostream>
#include "time.h"
#include <stdio.h>

using std::vector;
using std::string;

#pragma warning(disable:4996) //��ȡʱ�亯��ʧЧ����ȥ��

namespace FIMG
{

typedef std::vector<cv::Mat> Pyr_FEAT;

struct fusion_image_index //�ں�����ָ��
{
	double fusion_EN = 0.0;				//��
	double fusion_MI = 0.0;				//�ںϻ���Ϣ
	double fusion_meanGrad = 0.0;		//ƽ���ݶ�
	double fusion_SSIM_irf = 0.0;		//�ṹ���ƶ�
	double fusion_SSIM_visf = 0.0;		//
	double fusion_SF = 0.0;				//�ռ�Ƶ��
	double fusion_SD = 0.0;				//�Աȶ�
	double fusion_CC_irf = 0.0;			//���ϵ��
	double fusion_CC_visf = 0.0;
	double fusion_VIF = 0.0;			//�Ӿ���Ϣ�����
	double fusion_Qab_f = 0.0;			//��Ե��Ϣ������
};

struct affine_trans_params
{
	double x_scale = 1.0;		//x �����ű���
	double y_scale = 1.0;		//y �����ű���
	double rotate_angle = 0.0;	//��ת�Ƕȣ�degree����ֵΪ��ʱ����ת
	double x_offset = 0.0;		//x ��ƽ����
	double y_offset = 0.0;		//y ��ƽ����

};

struct Fusion_Image_Param
{
	int iPyramid_level = 4;
	double dAlpha = 0.7;
	double dGamma = 1.8;

	int iThresh_max_EN = 0;//����طָ��ʼ��ֵ
	int iMax_EN_p = 10;//����طָ����ֵ
	bool init_flag = false;

	affine_trans_params affine_param;
	fusion_image_index  image_indictor;

	//����궨��Ӧ�Ծ���
	cv::Mat homo_mat_LFuse = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 1, 1, 0, 0, 1); //��΢��+����	
	cv::Mat homo_mat_RFuse = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 1, 1, 0, 0, 1); //��΢��+����	
	cv::Mat homo_mat_LRVision = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 1, 1, 0, 0, 1); //����΢�������Ӿ���������

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


		int blockSize = 5; //����Ϊ��1������ һ��ȡ 5-21
		int minDisparity = 0; //��С�Ӳ�
		int numDisparities_ratio = 10;// numDisparities = 16 * ratio �ӲΧ*����
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
		int P1 = 600; //P1 = 8 * left.channels() * blockSize* blockSize ƽ��ϵ��
		int P2 = 2400;


		int blockSize = 5; //����Ϊ��1������ һ��ȡ 5-21
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