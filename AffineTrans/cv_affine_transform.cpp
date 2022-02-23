#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include "funcHead.h"
// ͼ�����任��ģ������
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
	double angle = 30;  //��ת�Ƕ�
	double tx = 50, ty = 100; //ƽ�ƾ���
	double cx = 1.5, cy = 1.5; //���ų߶�
	
	cv::Point2f zero_0(0.0, 0.0);
	cv::Point2f center = cv::Point2f(src.cols / 2, src.rows / 2);
	double Angle = angle * CV_PI / 180.0;

	double alpha = cx * cos(Angle);//cos��sin�Ի���Ϊ��λ
	double beta = cy * sin(Angle);

	

	clock_t time_s = clock();
	//���ű任
	//cv::Mat  mat_trans_c;
	//cv::Mat tempMat_c = (cv::Mat_<double>(2, 3) << cx, 0.0, 0.0, 0.0, cy, 0.0);
	//int dst_sc_rows = round(cy * src.rows);//�߶ȱ任��ͼ��߶�
	//int dst_sc_cols = round(cx * src.cols);//�߶ȱ任��ͼ����
	//warpAffine(src, mat_trans_c, tempMat_c, cv::Size(dst_sc_cols, dst_sc_rows));//���ŵĻ�׼��Ϊ���Ͻ�

	//��ת������
	cv::Mat mat_trans_r;
	double scale = 1.5;
	cv::Mat tempMat_r = cv::getRotationMatrix2D(center, angle, scale);//��ת��׼��center����ѡ��1Ϊ���ű�����angle�Ƕ�Ϊ��λ��
	int dst_sa_rows = round(fabs(scale * src.rows * cos(Angle)) + fabs(scale * src.cols * sin(Angle)));//�پ�����ת��ͼ��߶�
	int dst_sa_cols = round(fabs(scale * src.cols * cos(Angle)) + fabs(scale * src.rows * sin(Angle)));//�پ�����ת��ͼ����
	tempMat_r.at<double>(0, 2) += (dst_sa_cols - src.cols) / 2;//ƽ����ʾȫͼ
	tempMat_r.at<double>(1, 2) += (dst_sa_rows - src.rows) / 2;
	warpAffine(src, mat_trans_r, tempMat_r, cv::Size(dst_sa_cols, dst_sa_rows));//���ŵĻ�׼��Ϊ���Ͻ�,angleֵΪ������ʱ����ת
	
	//ƽ�Ʊ任
	// cv::Mat mat_trans_t;
	//cv::Mat tempMat_t = (cv::Mat_<double>(2, 3) << 1.0, 0.0, tx, 0.0, 1.0, ty);
	//warpAffine(src, mat_trans_t, tempMat_t, src.size());//����������ŵĻ�׼��Ϊ���Ͻ�
	

	//�ۺϱ任������任2*3
	cv::Mat all_mat = (cv::Mat_<double>(2, 3) << alpha, beta, ((1.0 - alpha) * center.x - beta * center.y) + tx,
												-beta, alpha, (beta * center.x + (1.0 - alpha) * center.y) + ty);

	int dst_rows = round(fabs(cy * src.rows * cos(Angle)) + fabs(cx * src.cols * sin(Angle)));//������ת��ͼ��߶�
	int dst_cols = round(fabs(cx * src.cols * cos(Angle)) + fabs(cy * src.rows * sin(Angle)));//������ת��ͼ����
	all_mat.at<double>(0, 2) += ((dst_cols - src.cols) / 2 - tx);//ƽ����ʾȫͼ
	all_mat.at<double>(1, 2) += ((dst_rows - src.rows) / 2 - ty);
	cv::Mat mat_trans_all;
	cv::Size ddd = cv::Size(dst_cols, dst_rows);
	warpAffine(src, mat_trans_all, all_mat, ddd);//���ŵĻ�׼��Ϊ���Ͻ�
	std::cout << all_mat << std::endl << std::endl;
	//warpPerspective(src,dst,homography,cv::Size());//͸�ӱ任3*3

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