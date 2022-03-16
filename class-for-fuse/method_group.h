#pragma once
#include "parameter.h"

#define MAX_PATH 256

namespace FIMG
{
class method_group
{
public:

	method_group();

	virtual ~method_group();

	//图像熵
	float entropy_a(cv::Mat& src);
	//图像联合熵
	float entropy_ab(cv::Mat& src1, cv::Mat& src2);

	//图像二维熵
	float entropy_2(cv::Mat& src);

	//图像互信息MI
	float multi_info(cv::InputArray& src1, cv::InputArray& src2, cv::InputArray& fusedimg);

	//图像平均梯度
	float mean_grad(cv::Mat& src);

	//SSIM结构化相似度
	float calc_SSIM(cv::Mat& src, cv::Mat& fuse);

	float calc_SF(cv::Mat& src);
	//相关系数
	float calc_CC(cv::Mat& src1, cv::Mat& src2);

	//画图像直方图
	void hist_graph(cv::Mat hist);

	//引导滤波，guidImg,引导图像,  input:输入图像,  r:核半径， eps:
	cv::Mat GuidedFilter(cv::Mat& input, cv::Mat& guidImg, int r, double eps);

	//最大熵分割算法--图像二值化
	int Max_Entropy(cv::Mat& src, cv::Mat& dst, int p = 10, int filter_size = 3);

	//平均灰度自适应gamma校正
	int Adjust_gamma(cv::Mat& src, cv::Mat& dst, float gamma = 1.0);

	//线性调节亮度
	int Adjust_contrast_brightness(cv::Mat& src, cv::Mat& dst, double alpha, double beta);

	//仿射变换
	int Affine_trans_base_matrix(cv::Mat& src, cv::Point2f trans_center, cv::Mat& dst, affine_trans_params& param, bool fullDisplayImg = false);

	//分块直方图均衡化
	void calc_clache_hist(cv::Mat& src, cv::Mat& dst, double dVlaue = 40.0, cv::Size img_block = cv::Size(8, 8));

	//线性融合方法: dAlpha * src1 + (1 - dAlpha) * src2
	int Alpha_Beta_Image_fuse(cv::Mat& src1, cv::Mat& src2, cv::Mat& dst, double dAlpha, double gamma = 0);

	//灰度拉伸
	int GrayStretch(cv::Mat& src, cv::Mat& dst, double dmin_s = 0.0, double dmax_s = 255.0);

	//显著特征图提取
	void SalientRegionDetectionBasedonLC(cv::Mat& src, cv::Mat& dst);

	//区域自适应生长分割
	void RegionGrow(cv::Mat src, cv::Mat& matDst);

	//获取当前文件路径
	string getCurFilePath();

	//文件类型 列表获取，仅适用于Windows,Linux 请注释
	void listFiles(string dir, vector<string>& files, string str_img_type);

	cv::Mat All_Images_inWindow(vector<cv::Mat> vct_img, int w_n = 2, int h_n = 2);

	void saveIndex2Txt(FIMG::fusion_image_index, string path = "");

public:

};
}
