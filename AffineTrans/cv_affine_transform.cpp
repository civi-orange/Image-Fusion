#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include "funcHead.h"
// 图像仿射变换的模板例子
int main3333()
{
	cv::Mat src = cv::imread("./image/2.jpg");
	resize(src, src, cv::Size(src.cols / 2, src.rows / 2));
	if (src.empty()) {
		std::cout << "Failure to load image..." << std::endl;
		return -1;
	}
	cvtColor(src, src, cv::COLOR_BGR2GRAY);

	cv::Mat dst;
	double angle = 30;  //旋转角度
	double tx = 50, ty = 100; //平移距离
	double cx = 1.5, cy = 1.5; //缩放尺度
	
	cv::Point2f zero_0(0.0, 0.0);
	cv::Point2f center = cv::Point2f(src.cols / 2, src.rows / 2);
	double Angle = angle * CV_PI / 180.0;

	double alpha = cx * cos(Angle);//cos，sin以弧度为单位
	double beta = cy * sin(Angle);

	

	clock_t time_s = clock();
	//缩放变换
	//cv::Mat  mat_trans_c;
	//cv::Mat tempMat_c = (cv::Mat_<double>(2, 3) << cx, 0.0, 0.0, 0.0, cy, 0.0);
	//int dst_sc_rows = round(cy * src.rows);//尺度变换后图像高度
	//int dst_sc_cols = round(cx * src.cols);//尺度变换后图像宽度
	//warpAffine(src, mat_trans_c, tempMat_c, cv::Size(dst_sc_cols, dst_sc_rows));//缩放的基准点为左上角

	//旋转，缩放
	cv::Mat mat_trans_r;
	double scale = 1.5;
	cv::Mat tempMat_r = cv::getRotationMatrix2D(center, angle, scale);//旋转基准点center可以选择，1为缩放倍数，angle是度为单位，
	int dst_sa_rows = round(fabs(scale * src.rows * cos(Angle)) + fabs(scale * src.cols * sin(Angle)));//再经过旋转后图像高度
	int dst_sa_cols = round(fabs(scale * src.cols * cos(Angle)) + fabs(scale * src.rows * sin(Angle)));//再经过旋转后图像宽度
	tempMat_r.at<double>(0, 2) += (dst_sa_cols - src.cols) / 2;//平移显示全图
	tempMat_r.at<double>(1, 2) += (dst_sa_rows - src.rows) / 2;
	warpAffine(src, mat_trans_r, tempMat_r, cv::Size(dst_sa_cols, dst_sa_rows));//缩放的基准点为左上角,angle值为正：逆时针旋转
	
	//平移变换
	// cv::Mat mat_trans_t;
	//cv::Mat tempMat_t = (cv::Mat_<double>(2, 3) << 1.0, 0.0, tx, 0.0, 1.0, ty);
	//warpAffine(src, mat_trans_t, tempMat_t, src.size());//这个方法缩放的基准点为左上角
	

	//综合变换，仿射变换2*3
	cv::Mat all_mat = (cv::Mat_<double>(2, 3) << alpha, beta, ((1.0 - alpha) * center.x - beta * center.y) + tx,
												-beta, alpha, (beta * center.x + (1.0 - alpha) * center.y) + ty);

	int dst_rows = round(fabs(cy * src.rows * cos(Angle)) + fabs(cx * src.cols * sin(Angle)));//经过旋转后图像高度
	int dst_cols = round(fabs(cx * src.cols * cos(Angle)) + fabs(cy * src.rows * sin(Angle)));//经过旋转后图像宽度
	all_mat.at<double>(0, 2) += ((dst_cols - src.cols) / 2 - tx);//平移显示全图
	all_mat.at<double>(1, 2) += ((dst_rows - src.rows) / 2 - ty);
	cv::Mat mat_trans_all;
	cv::Size ddd = cv::Size(dst_cols, dst_rows);
	warpAffine(src, mat_trans_all, all_mat, ddd);//缩放的基准点为左上角
	std::cout << all_mat << std::endl << std::endl;
	//warpPerspective(src,dst,homography,cv::Size());//透视变换3*3

	clock_t time_e = clock();
	std::cout << "running time: " << time_e - time_s << "ms" << std::endl;
	//imshow("mat_trans_c", mat_trans_c);
	imshow("mat_trans_r", mat_trans_r);
	//imshow("mat_trans_t", mat_trans_t);
	imshow("mat_trans_all", mat_trans_all);
	
	//just for test the function
	//affine_trans_params fffffff;
	//fffffff.rotate_angle = angle;
	//fffffff.x_offset = tx;
	//fffffff.y_offset = ty;
	//fffffff.x_scale = cx;
	//fffffff.y_scale = cy;
	//cv::Mat outtttt;
	//Affine_trans_base_matrix(src, center, fffffff, outtttt, false);
	//imshow("outtttt", outtttt);
	//Affine_trans_base_matrix(src, center, fffffff, outtttt, true);
	//imshow("outtttt1", outtttt);


	cv::waitKey();

	return 0;
}