#include "stereo_match_group.h"

SMAG::stereo_match_group::stereo_match_group()
{

}

SMAG::stereo_match_group::~stereo_match_group()
{

}

void SMAG::stereo_match_group::setBMParam(SMAG::BM_param& parameter)
{
	bm_param = parameter.clone();
}

void SMAG::stereo_match_group::setSGBMParam(SMAG::SGBM_param& parameter)
{
	sgbm_param = parameter.clone();
}

int SMAG::stereo_match_group::BM_stereo(cv::Mat& left, cv::Mat& right, cv::Mat& disparity)
{
	cv::Mat left1, right1;
	if (left.channels() > 1)
	{
		cvtColor(left, left1, cv::COLOR_BGR2GRAY);
	}
	if (right.channels() > 1)
	{
		cvtColor(right, right1, cv::COLOR_BGR2GRAY);
	}
	cv::Ptr<cv::StereoBM> BMState = cv::StereoBM::create();
	assert(BMState);

	BMState->setPreFilterType(bm_param.preFiletertype);
	BMState->setPreFilterSize(bm_param.preFilterSize);
	BMState->setPreFilterCap(bm_param.preFileterCap);
	int blockSize = bm_param.blockSize;
	if (blockSize % 2 == 0) { blockSize += 1; }
	if (blockSize < 5) { blockSize = 5; }
	BMState->setBlockSize(blockSize);

	BMState->setMinDisparity(bm_param.minDisparity);
	BMState->setNumDisparities(bm_param.numDisparities_ratio);
	BMState->setTextureThreshold(bm_param.textureThreshold);
	BMState->setUniquenessRatio(bm_param.UniquenessRatio);
	BMState->setSpeckleWindowSize(bm_param.speckleWindowSize);
	BMState->setSpeckleRange(bm_param.SpeckleRange);
	BMState->setDisp12MaxDiff(bm_param.disp12MaxDiff);
	BMState->setROI1(bm_param.roi_left);
	BMState->setROI2(bm_param.roi_right);

	BMState->compute(left1, right1, disparity);

	return true;
}

int SMAG::stereo_match_group::SGBM_stereo(cv::Mat& left, cv::Mat& right, cv::Mat& disparity)
{
	cv::Ptr<cv::StereoSGBM> SGBM = cv::StereoSGBM::create();

	SGBM->setMode(sgbm_param.mode);
	SGBM->setPreFilterCap(sgbm_param.preFileterCap);
	SGBM->setMinDisparity(sgbm_param.minDisparity);

	int blockSize = sgbm_param.blockSize;
	if (blockSize % 2 == 0) { blockSize += 1; }
	if (blockSize < 5) { blockSize = 5; }
	SGBM->setBlockSize(blockSize);
	sgbm_param.P1 = 8 * blockSize * blockSize * left.channels();
	sgbm_param.P2 = 4 * sgbm_param.P1;
	SGBM->setP1(sgbm_param.P1);
	SGBM->setP2(sgbm_param.P2);//P2 = 4 * P1

	SGBM->setNumDisparities(sgbm_param.numDisparities_ratio);
	SGBM->setSpeckleWindowSize(sgbm_param.speckleWindowSize);
	SGBM->setSpeckleRange(sgbm_param.SpeckleRange);
	SGBM->setUniquenessRatio(sgbm_param.UniquenessRatio);
	SGBM->setDisp12MaxDiff(sgbm_param.disp12MaxDiff);

	SGBM->compute(left, right, disparity);

	return true;
}

//后处理
void SMAG::stereo_match_group::insertDepth32f(cv::Mat& src_32f, cv::Mat& dst_32f)
{
	const int width = src_32f.cols;
	const int height = src_32f.rows;
	float* data = (float*)src_32f.data;
	cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
	cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
	double* integral = (double*)integralMap.data;
	int* ptsIntegral = (int*)ptsMap.data;
	memset(integral, 0, sizeof(double) * width * height);
	memset(ptsIntegral, 0, sizeof(int) * width * height);
	for (int i = 0; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 0; j < width; ++j)
		{
			int id2 = id1 + j;
			if (data[id2] > 1e-3)
			{
				integral[id2] = data[id2];
				ptsIntegral[id2] = 1;
			}
		}
	}
	// 积分区间
	for (int i = 0; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 1; j < width; ++j)
		{
			int id2 = id1 + j;
			integral[id2] += integral[id2 - 1];
			ptsIntegral[id2] += ptsIntegral[id2 - 1];
		}
	}
	for (int i = 1; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 0; j < width; ++j)
		{
			int id2 = id1 + j;
			integral[id2] += integral[id2 - width];
			ptsIntegral[id2] += ptsIntegral[id2 - width];
		}
	}
	int wnd;
	double dWnd = 2;
	while (dWnd > 1)
	{
		wnd = int(dWnd);
		dWnd /= 2;
		for (int i = 0; i < height; ++i)
		{
			int id1 = i * width;
			for (int j = 0; j < width; ++j)
			{
				int id2 = id1 + j;
				int left = j - wnd - 1;
				int right = j + wnd;
				int top = i - wnd - 1;
				int bot = i + wnd;
				left = max(0, left);
				right = min(right, width - 1);
				top = max(0, top);
				bot = min(bot, height - 1);
				int dx = right - left;
				int dy = (bot - top) * width;
				int idLeftTop = top * width + left;
				int idRightTop = idLeftTop + dx;
				int idLeftBot = idLeftTop + dy;
				int idRightBot = idLeftBot + dx;
				int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
				double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
				if (ptsCnt <= 0)
				{
					continue;
				}
				data[id2] = float(sumGray / ptsCnt);
			}
		}
		int s = wnd / 2 * 2 + 1;
		if (s > 201)
		{
			s = 201;
		}
		cv::GaussianBlur(src_32f, dst_32f, cv::Size(s, s), s, s);
	}
}

int SMAG::stereo_match_group::dispaly(cv::Mat& src, cv::Mat& dst)
{
	//src.convertTo(dst, CV_8UC1, 255 / ((sgbm_param.numDisparities_ratio * 16) * 16.));//normalize 计算公式

	normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	imshow("disparity image", dst);
	cv::waitKey(1);
	return true;
}
