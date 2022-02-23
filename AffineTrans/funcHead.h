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

#include<Python.h>
#include "D:/Anaconda3/pkgs/numpy-base-1.20.3-py39hc2deb75_0/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"

using namespace std;

#define METHOD_SURF_DETECTOR 0X00000000A0001
#define METHOD_SIFT_DETECTOR 0X00000000A0002
#define METHOD_FAST_DETECTOR 0X00000000A0003
#define MAX_PATH 256
#pragma warning(disable:4996)
typedef struct
{
	cv::Point2f left_top;
	cv::Point2f left_bottom;
	cv::Point2f right_top;
	cv::Point2f right_bottom;
}four_corners_t;


struct fusion_image_index
{
	double fusion_EN = 0.0;//��
	double fusion_MI = 0.0;//�ںϻ���Ϣ
	double fusion_meanGrad = 0.0;//ƽ���ݶ�
	double fusion_SSIM_irf = 0.0;//�ṹ���ƶ�
	double fusion_SSIM_visf = 0.0;//
	double fusion_SF = 0.0;//�ռ�Ƶ��
	double fusion_SD = 0.0;//�Աȶ�
	double fusion_CC_irf = 0.0;//���ϵ��
	double fusion_CC_visf = 0.0;
	double fusion_VIF = 0.0;//�Ӿ���Ϣ�����
	double fusion_Qab_f= 0.0;//��Ե��Ϣ������
};

struct affine_trans_params
{
	double x_scale;//x�����ű���
	double y_scale;//y�����ű���
	double rotate_angle;//��ת�Ƕȣ�degree����ֵΪ��ʱ����ת
	double x_offset;//x��ƽ����
	double y_offset;//y��ƽ����
	affine_trans_params()
	{
		x_scale = 1.0;
		y_scale = 1.0;
		rotate_angle = 0.0;
		x_offset = 0.0;
		y_offset = 0.0;
	}
};

void listFiles(string dir, vector<string>& files, string str_img_type);

//��˹����������ͼ
std::vector<cv::Mat> Gauss_Pyr(cv::Mat& input, std::vector<cv::Mat>& Img_pyr, int level);//��˹������

//������˹����������ͼ
std::vector<cv::Mat> Laplace_Pyr(std::vector<cv::Mat>& Img_Gaussian_pyr, std::vector<cv::Mat>& Img_Laplacian_pyr, int level);//������˹������

//������˹�������ں�
std::vector<cv::Mat> Fusion_Laplace_Pyr(std::vector<cv::Mat>& Img_front_lp, std::vector<cv::Mat>& Img_back_lp, std::vector<cv::Mat>& mask_gau, std::vector<cv::Mat>& blend_lp);//�������ں�
																																												
//ͼ���ں�
int Laplace_pyramid_fusion(cv::Mat& img_back, cv::Mat& img_front, double alpha, int pyr_level, cv::Mat& output);//ͼ����ӻ�ԭ

//ͼ���������ں�
int ImageMatch(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo, int detectorType);

int fast_Detector(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo);

int surf_Detector(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo);

int sift_Detector(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo);

//͸�ӱ任�ǵ����
void CalcCorners(const cv::Mat& H, const cv::Mat& src, four_corners_t& corners);//��H*SRC����ͼ�ǵ�����

//���������ƽ��ƴ�Ӳ���
void OptimizeSeam(cv::Mat& img1, cv::Mat& trans, four_corners_t& corners, cv::Mat& dst);//�ںϽӿ�

//�Ҷȼ�ͳ��
void CountHistNum(cv::Mat& src, std::vector<int>& out);

//���������ָ�
void RegionGrow(cv::Mat src, cv::Mat& matDst);

//��ȡ���������ߵ�����
int Get_Infrared_target_region(cv::Mat& src, cv::Mat& dst);

//��������ͼ��ȡ
void SalientRegionDetectionBasedonLC(cv::Mat& src, cv::Mat& dst);

int GrayStretch(cv::Mat& src, cv::Mat& dst, double dmin_s = 0.0, double dmax_s = 255.0);

//�����ںϷ���: dAlpha * src1 + (1 - dAlpha) * src2
int Alpha_Beta_Image_fuse(cv::Mat& src1, cv::Mat& src2, cv::Mat& dst,  double dAlpha, double gamma = 0);

void calc_clache_hist(cv::Mat& src, cv::Mat& dst, double dValue = 40.0, cv::Size img_block = cv::Size(8, 8));

//����任
int Affine_trans_base_matrix(cv::Mat& src, cv::Point2f& trans_center, affine_trans_params& param, cv::Mat& dst, bool fullDisplayImg);

//���Ե�������
int Adjust_contrast_brightness(cv::Mat& src, cv::Mat& dst, double alpha, double beta);

int Adjust_gamma(cv::Mat& src, cv::Mat& dst, float gamma = 1.0);

//ͼ����
float entropy_a(cv::Mat& src);
//ͼ��������
float entropy_ab(cv::Mat& src1, cv::Mat& src2);

//ͼ���ά��
float entropy_2(cv::Mat& src);

//ͼ����ϢMI
float multi_info(cv::InputArray& src1, cv::InputArray& src2, cv::InputArray& fusedimg);

//ͼ��ƽ���ݶ�
float mean_grad(cv::Mat& src);


//SSIM�ṹ�����ƶ�
float calc_SSIM(cv::Mat& src, cv::Mat& fuse);

float calc_SF(cv::Mat& src);

float calc_CC(cv::Mat& src1, cv::Mat& src2);

void hist_graph(cv::Mat hist);

int image_fusion_evalution(cv::Mat frame_IR, cv::Mat frame_vis, cv::Mat frame_fuse, fusion_image_index& output);

//�����˲���guidImg,����ͼ��,  input:����ͼ��,  r:�˰뾶�� eps:
cv::Mat GuidedFilter( cv::Mat& input, cv::Mat& guidImg, int r, double eps);

cv::Mat All_Images_inWindow(vector<cv::Mat> vct_img);

string getCurFilePath();

//����طָ��㷨--ͼ���ֵ��
int Max_Entropy(cv::Mat& src, cv::Mat& dst, int thresh = 0, int p = 10);
