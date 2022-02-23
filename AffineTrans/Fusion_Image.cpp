#include "Fusion_Image.h"

FIMG::Fusion_Image::Fusion_Image()
{
	
}

FIMG::Fusion_Image::~Fusion_Image()
{

}

int FIMG::Fusion_Image::init_parameter(const FIMG::Fusion_Image_Param& parameter)
{
	param_.affine_param = parameter.affine_param;
	param_.dAlpha = parameter.dAlpha;
	param_.dGamma = parameter.dGamma;
	param_.image_indictor = parameter.image_indictor;
	param_.iPyramid_level = parameter.iPyramid_level;
	param_.iMax_EN_p = parameter.iMax_EN_p;
	param_.iThresh_max_EN = parameter.iThresh_max_EN;

	return true;
}


int FIMG::Fusion_Image::init_fusion(const cv::Mat& img_std)
{
	//According to the initialization, set the image pyramid layer maximum fit 
	int temp_int = std::pow(2, param_.iPyramid_level);
	//the redundant code
	while (img_std.cols % temp_int != 0 || img_std.rows % temp_int != 0)
	{
		//std::cout<<"The pyramid has too many layers!"<< std::endl;
		param_.iPyramid_level -= 1;
		temp_int = pow(2, param_.iPyramid_level);
	}

	feature_.SetParam(param_);

	cv::Mat mask = cv::Mat::zeros(img_std.size(), CV_32FC1);
	mask(cv::Range::all(), cv::Range::all()) = param_.dAlpha;//多尺度线性融合
	if (img_std.channels() == 3)
	{ 
		cvtColor(mask, mask, cv::COLOR_GRAY2BGR); 
	}
	feature_.Gauss_Pyr(mask, mask_Pyr_);

	return true;
}

int FIMG::Fusion_Image::fusion_image(const cv::Mat& img_front, const cv::Mat& img_back, cv::Mat& img_dst)
{
	front_Gauss_Pyr_.clear();
	front_Laplace_Pyr_.clear();
	back_Gauss_Pyr_.clear();
	back_Laplace_Pyr_.clear();

	FIMG::Pyr_FEAT out_temp;
	cv::Mat img_front_f, img_back_f;
	img_front.convertTo(img_front_f, CV_32F);
	img_back.convertTo(img_back_f, CV_32F);

	feature_.Gauss_Pyr(img_front_f, front_Gauss_Pyr_);
	feature_.Gauss_Pyr(img_back_f, back_Gauss_Pyr_);
	feature_.Laplace_Pyr(front_Gauss_Pyr_, front_Laplace_Pyr_);
	feature_.Laplace_Pyr(back_Gauss_Pyr_, back_Laplace_Pyr_);

	laplace_pyr_fusion(front_Laplace_Pyr_, back_Laplace_Pyr_, mask_Pyr_, out_temp);

	cv::Mat img_up;
	cv::Mat img_start = front_Gauss_Pyr_.back().mul(mask_Pyr_.back()) + ((back_Gauss_Pyr_.back()).mul(cv::Scalar(1.0) - mask_Pyr_.back()));//灰度图
	int l = out_temp.size();
	for (int i = 0; i < l; i++)
	{
		pyrUp(img_start, img_up, cv::Size(img_start.cols * 2, img_start.rows * 2));
		img_dst = out_temp[l - i - 1] + img_up;
		img_start = img_dst;
	}
	img_dst.convertTo(img_dst, CV_8UC1);

	return true;
}


int FIMG::Fusion_Image::Get_Fusion_indicator(cv::Mat img_front, cv::Mat img_back, cv::Mat img_fuse, fusion_image_index& fuse_indicator)
{
	if (img_front.empty() || img_back.empty() || img_fuse.empty())
	{
		return false;
	}
	//EN
	fuse_indicator.fusion_EN = method_.entropy_a(img_fuse);

	//MI
	fuse_indicator.fusion_MI = method_.multi_info(img_front, img_back, img_fuse);
	//mean grad
	fuse_indicator.fusion_meanGrad = method_.mean_grad(img_fuse);
	//SSIM
	fuse_indicator.fusion_SSIM_irf = method_.calc_SSIM(img_front, img_fuse);
	fuse_indicator.fusion_SSIM_visf = method_.calc_SSIM(img_back, img_fuse);
	//SF
	fuse_indicator.fusion_SF = method_.calc_SF(img_fuse);
	//CC
	fuse_indicator.fusion_CC_irf = method_.calc_CC(img_front, img_fuse);
	fuse_indicator.fusion_CC_visf = method_.calc_CC(img_back, img_fuse);
	//SD
	cv::Mat mean_val, sd_val;
	meanStdDev(img_fuse, mean_val, sd_val);
	fuse_indicator.fusion_SD = sd_val.at<double>(0, 0);
	//Q_ab/f
	fuse_indicator.fusion_Qab_f = -1;//未实现
	//VIF
	fuse_indicator.fusion_VIF = -1;//未实现

	return true;

}



int FIMG::Fusion_Image::laplace_pyr_fusion(FIMG::Pyr_FEAT& Img_lp_front, FIMG::Pyr_FEAT& Img_lp_back, FIMG::Pyr_FEAT& mask_gau, FIMG::Pyr_FEAT& blend_lp)
{

	int level = Img_lp_front.size();

	for (int i = 0; i < level; i++)//0为大图即金字塔底层                                        
	{
		int xxx = Img_lp_front[i].type();
		int xxx1 = mask_gau[i].type();
		cv::Mat A = Img_lp_front[i].mul(mask_gau[i]); //Alpha（Attention:）Data types must be unified 
		cv::Mat antiMask;
		if (mask_gau[0].channels() == 3) {
			antiMask = cv::Scalar(1.0, 1.0, 1.0) - mask_gau[i];//彩色图
		}
		else {
			antiMask = cv::Scalar(1.0) - mask_gau[i];//灰度图
		}
		cv::Mat B = Img_lp_back[i].mul(antiMask); //Beta
		cv::Mat blendedLevel = A + B;
		blend_lp.push_back(blendedLevel);
	}

	return true;
}


