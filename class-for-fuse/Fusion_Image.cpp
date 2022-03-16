#include "Fusion_Image.h"

FIMG::Fusion_Image::Fusion_Image()
{
	vct_guid_45.reserve(2);
	vct_guid_7.reserve(2);
}

FIMG::Fusion_Image::~Fusion_Image()
{
	vct_guid_45.clear();
	vct_guid_7.clear();
	mask_Pyr_.clear();
	front_Gauss_Pyr_.clear();
	front_Laplace_Pyr_.clear();
	back_Gauss_Pyr_.clear();
	back_Laplace_Pyr_.clear();
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
	feature_.SetParam(param_);//Obtain the maximum number of pyramid layers without changing the size

	
	cv::Mat mask = cv::Mat::zeros(img_std.size(), CV_32FC1);
	mask(cv::Range::all(), cv::Range::all()) = param_.dAlpha;//多尺度线性融合
	
	
	if (img_std.channels() == 3)
	{ 
		cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
	}

	feature_.Gauss_Pyr(mask, mask_Pyr_);

	return true;
}

int FIMG::Fusion_Image::fusion_image(const cv::Mat& img_front, const cv::Mat& img_back, cv::Mat& img_dst, cv::InputArray mask)
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

int FIMG::Fusion_Image::GFF_Fusion(const cv::Mat& img_front, const cv::Mat& img_back, cv::Mat& img_dst)
{
	if (img_front.empty() || img_back.empty())
	{
		return false;
	}
	cv::Mat mat_front, mat_back;
	img_front.copyTo(mat_front);
	img_back.copyTo(mat_back);

	get_fusion_weights(mat_front, mat_back);

	img_dst = cv::Mat::zeros(img_front.size(), CV_32F);

	std::vector<cv::Mat> vct_filter_mat;
	std::vector<cv::Mat> vct_diff_mat;

	mat_front.convertTo(mat_front, CV_32F);
	mat_front.convertTo(mat_back, CV_32F);
	cv::Mat filter_mat, diff_mat;
	int size_b = 13;
	boxFilter(mat_back, filter_mat, -1, cv::Size(size_b, size_b));
	diff_mat = mat_back - filter_mat;
	vct_filter_mat.push_back(filter_mat);
	vct_diff_mat.push_back(diff_mat);
	boxFilter(mat_front, filter_mat, -1, cv::Size(size_b, size_b));
	diff_mat = mat_front - filter_mat;
	vct_filter_mat.push_back(filter_mat);
	vct_diff_mat.push_back(diff_mat);
	

	if (img_front.channels() == 3) 
	{
		std::vector<cv::Mat> vec;
		for (size_t i = 0; i < vct_guid_45.size(); ++i) 
		{
			vec.clear();
			for (int it = 0; it < 3; it++) 
			{
				vec.push_back(vct_guid_45[i]);
			}
			cv::merge(vec, vct_guid_45[i]);
			vec.clear();
			for (int it = 0; it < 3; it++) 
			{
				vec.push_back(vct_guid_7[i]);
			}
			cv::merge(vec, vct_guid_7[i]);
		}
	}

	for (size_t i = 0; i < vct_guid_45.size(); ++i) {
		cv::Mat temp1 = vct_guid_45[i].mul(vct_filter_mat[i]);
		cv::Mat temp2 = vct_guid_7[i].mul(vct_diff_mat[i]);
		img_dst = img_dst + temp1 + temp2;
	}
	img_dst.convertTo(img_dst, CV_8U);
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

int FIMG::Fusion_Image::LaplacianOfTheImage(const cv::Mat& src, cv::Mat& dst, bool is_abs /*= true*/)
{
	cv::Mat laplaceFilter = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

	filter2D(src, dst, -1, laplaceFilter);

	if (is_abs == true)
	{
		dst = cv::abs(dst);
	}
	return true;
}


int FIMG::Fusion_Image::get_fusion_weights(const cv::Mat& src_front, const cv::Mat& src_back)
{
	vct_guid_45.clear();
	vct_guid_7.clear();
	vector<cv::Mat> vct_weights;
	vct_weights.reserve(2);	

	cv::Mat lp_front, lp_back;
	LaplacianOfTheImage(src_front, lp_front);
	LaplacianOfTheImage(src_front, lp_back);

	for (int it = 0; it < 2; ++it)
	{
		//vct_weights[0] : vis weights, vct_weights[1] : IR weights
		vct_weights.push_back(cv::Mat::zeros(src_front.size(), CV_32FC1));
	}

	for (int x = 0; x < src_front.rows; ++x)
	{
		for (int y = 0; y < src_front.cols; ++y)
		{
			for (int i = 0; i < 2; ++i) {
				if (lp_back.at<float>(x, y) <= lp_front.at<float>(x, y))
				{
					vct_weights[1].at<float>(x, y) = 1.0f;
				}
				else
				{
					vct_weights[0].at<float>(x, y) = 1.0f;
				}
			}
			
		}
	}



	cv::Mat guidf_mat;
	cv::ximgproc::guidedFilter(src_back, vct_weights[0], guidf_mat, 45, 0.3 * 255 * 255);
	vct_guid_45.push_back(guidf_mat);

	cv::ximgproc::guidedFilter(src_back, vct_weights[0], guidf_mat, 7, 1e-6 * 255 * 255);
	vct_guid_7.push_back(guidf_mat);

	cv::ximgproc::guidedFilter(src_front, vct_weights[1], guidf_mat, 45, 0.3 * 255 * 255);
	vct_guid_45.push_back(guidf_mat);

	cv::ximgproc::guidedFilter(src_front, vct_weights[1], guidf_mat, 7, 1e-6 * 255 * 255);
	vct_guid_7.push_back(guidf_mat);


	for (int x = 0; x < src_front.rows; ++x) 
	{
		for (int y = 0; y < src_front.cols; ++y) 
		{
			float sumB = 0.0f, sumD = 0.0f;

			for (size_t i = 0; i < vct_guid_45.size(); ++i)
			{
				float fB = vct_guid_45[i].at<float>(x, y);
				if (fB > 1.0f) 
				{
					vct_guid_45[i].at<float>(x, y) = 1.0f;
					fB = 1.0f;
				}
				sumB += fB;
				float fD = vct_guid_7[i].at<float>(x, y);
				if (fD > 1.0f) 
				{
					vct_guid_7[i].at<float>(y, x) = 1.0f;
					fD = 1.0f;
				}
				sumD += fD;
			}

			for (size_t i = 0; i < vct_guid_45.size(); ++i) {
				vct_guid_45[i].at<float>(x, y) /= sumB;
				vct_guid_7[i].at<float>(x, y) /= sumD;
			}
		}
	}

	return true;
}

