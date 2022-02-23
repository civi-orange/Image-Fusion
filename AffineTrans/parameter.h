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

};


}