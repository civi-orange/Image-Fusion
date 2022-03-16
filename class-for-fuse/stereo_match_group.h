#pragma once
#include "parameter.h"
#include "method_group.h"

using namespace std;

namespace SMAG
{

class stereo_match_group
{
public:

	stereo_match_group();

	virtual ~stereo_match_group();

	void setBMParam(SMAG::BM_param& parameter);

	void setSGBMParam(SMAG::SGBM_param& parameter);

	int BM_stereo(cv::Mat& left, cv::Mat& right, cv::Mat& disparity);

	int SGBM_stereo(cv::Mat& left, cv::Mat& right, cv::Mat& disparity);

	int dispaly(cv::Mat& src, cv::Mat& dst);

private:
	//∫Û¥¶¿Ì
	void insertDepth32f(cv::Mat& src_32f, cv::Mat& dst_32f);



public:
	BM_param bm_param;
	SGBM_param sgbm_param;

};

}

