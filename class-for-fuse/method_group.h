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
	//���ϵ��
	float calc_CC(cv::Mat& src1, cv::Mat& src2);

	//��ͼ��ֱ��ͼ
	void hist_graph(cv::Mat hist);

	//�����˲���guidImg,����ͼ��,  input:����ͼ��,  r:�˰뾶�� eps:
	cv::Mat GuidedFilter(cv::Mat& input, cv::Mat& guidImg, int r, double eps);

	//����طָ��㷨--ͼ���ֵ��
	int Max_Entropy(cv::Mat& src, cv::Mat& dst, int p = 10, int filter_size = 3);

	//ƽ���Ҷ�����ӦgammaУ��
	int Adjust_gamma(cv::Mat& src, cv::Mat& dst, float gamma = 1.0);

	//���Ե�������
	int Adjust_contrast_brightness(cv::Mat& src, cv::Mat& dst, double alpha, double beta);

	//����任
	int Affine_trans_base_matrix(cv::Mat& src, cv::Point2f trans_center, cv::Mat& dst, affine_trans_params& param, bool fullDisplayImg = false);

	//�ֿ�ֱ��ͼ���⻯
	void calc_clache_hist(cv::Mat& src, cv::Mat& dst, double dVlaue = 40.0, cv::Size img_block = cv::Size(8, 8));

	//�����ںϷ���: dAlpha * src1 + (1 - dAlpha) * src2
	int Alpha_Beta_Image_fuse(cv::Mat& src1, cv::Mat& src2, cv::Mat& dst, double dAlpha, double gamma = 0);

	//�Ҷ�����
	int GrayStretch(cv::Mat& src, cv::Mat& dst, double dmin_s = 0.0, double dmax_s = 255.0);

	//��������ͼ��ȡ
	void SalientRegionDetectionBasedonLC(cv::Mat& src, cv::Mat& dst);

	//��������Ӧ�����ָ�
	void RegionGrow(cv::Mat src, cv::Mat& matDst);

	//��ȡ��ǰ�ļ�·��
	string getCurFilePath();

	//�ļ����� �б��ȡ����������Windows,Linux ��ע��
	void listFiles(string dir, vector<string>& files, string str_img_type);

	cv::Mat All_Images_inWindow(vector<cv::Mat> vct_img, int w_n = 2, int h_n = 2);

	void saveIndex2Txt(FIMG::fusion_image_index, string path = "");

public:

};
}
