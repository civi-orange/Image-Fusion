// 灰度直方图拉伸示例
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// 使用Rect绘制直方图
void drawHist_Rect(const cv::Mat& hist, cv::Mat& canvas, const cv::Scalar& color)
{
	CV_Assert(!hist.empty() && hist.cols == 1);
	CV_Assert(hist.depth() == CV_32F && hist.channels() == 1);
	CV_Assert(!canvas.empty() && canvas.cols >= hist.rows);

	const int width = canvas.cols;
	const int height = canvas.rows;

	// 获取最大值
	double dMax = 0.0;
	cv::minMaxLoc(hist, nullptr, &dMax);

	// 计算直线的宽度
	float thickness = float(width) / float(hist.rows);

	// 绘制直方图
	for (int i = 1; i < hist.rows; ++i)
	{
		double h = hist.at<float>(i, 0) / dMax * 0.9 * height; // 最高显示为画布的90%
		cv::rectangle(canvas,
			cv::Point(static_cast<int>((i - 1) * thickness), height),
			cv::Point(static_cast<int>(i * thickness), static_cast<int>(height - h)),
			color,
			static_cast<int>(thickness));
	}
}

// 直方图拉伸
// grayImage - 要拉伸的单通道灰度图像
// hist - grayImage的直方图
// minValue - 忽略像数个数小于此值的灰度级
void histStretch(cv::Mat& grayImage, const cv::Mat& hist, int minValue)
{
	CV_Assert(!grayImage.empty() && grayImage.channels() == 1 && grayImage.depth() == CV_8U);
	CV_Assert(!hist.empty() && hist.rows == 256 && hist.cols == 1 && hist.depth() == CV_32F);
	CV_Assert(minValue >= 0);

	// 求左边界
	uchar grayMin = 0;
	for (int i = 0; i < hist.rows; ++i)
	{
		if (hist.at<float>(i, 0) > minValue)
		{
			grayMin = static_cast<uchar>(i);
			break;
		}
	}

	// 求右边界
	uchar grayMax = 0;
	for (int i = hist.rows - 1; i >= 0; --i)
	{
		if (hist.at<float>(i, 0) > minValue)
		{
			grayMax = static_cast<uchar>(i);
			break;
		}
	}

	if (grayMin >= grayMax)
	{
		return;
	}

	const int w = grayImage.cols;
	const int h = grayImage.rows;
	for (int y = 0; y < h; ++y)
	{
		uchar* imageData = grayImage.ptr<uchar>(y);
		for (int x = 0; x < w; ++x)
		{
			if (imageData[x] < grayMin)
			{
				imageData[x] = 0;
			}
			else if (imageData[x] > grayMax)
			{
				imageData[x] = 255;
			}
			else
			{
				imageData[x] = static_cast<uchar>(std::round((imageData[x] - grayMin) * 255.0 / (grayMax - grayMin)));
			}
		}
	}
}

int main68554565()
{
	// 读入图像，此时是3通道的RGB图像
	cv::Mat image = cv::imread("./image/VIS_18dhvR.bmp");
	if (image.empty())
	{
		return -1;
	}

	// 转换为单通道的灰度图
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	// 计算直方图并绘制
	cv::Mat hist;
	cv::Mat histCanvas(400, 512, CV_8UC3, cv::Scalar(255, 255, 255));
	int channels[1] = { 0 };
	int histSize = 256;
	float range[2] = { 0, 256 };
	const float* ranges[1] = { range };
	cv::calcHist(&grayImage, 1, channels, cv::Mat(), hist, 1, &histSize, ranges);
	drawHist_Rect(hist, histCanvas, cv::Scalar(255, 0, 0));

	// 显示原始灰度图像及其直方图
	cv::imshow("Gray image", grayImage);
	cv::imshow("Gray image's histogram", histCanvas);

	// 直方图拉伸
	cv::Mat grayImageStretched = grayImage.clone();
	histStretch(grayImageStretched, hist, 160);

	// 计算直方图并绘制
	cv::Mat histStretched;
	cv::Mat histCanvasStretched(400, 512, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::calcHist(&grayImageStretched, 1, channels, cv::Mat(), histStretched, 1, &histSize, ranges);
	drawHist_Rect(histStretched, histCanvasStretched, cv::Scalar(255, 0, 0));

	// 显示拉伸后的灰度图像及其直方图
	cv::imshow("Stretched image", grayImageStretched);
	cv::imshow("Stretched image's histogram", histCanvasStretched);

	cv::waitKey();
	return 0;
}
