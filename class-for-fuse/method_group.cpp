#include "method_group.h"

FIMG::method_group::method_group()
{

}

FIMG::method_group::~method_group()
{

}

float FIMG::method_group::entropy_a(cv::Mat& src)
{
	//ͼ��һά��
	cv::Mat hist;
	const int histSize[] = { 256 }; //ֱ��ͼÿһ��ά�Ȼ��ֵ���������Ŀ
	float pranges[] = { 0,256 };
	const float* ranges[] = { pranges };
	calcHist(&src, 1, 0, cv::Mat(), hist, 1, histSize, ranges, true, false);//����ֱ��ͼ	
	double _1_wh = 1.0 / ((long long)src.cols * (long long)src.rows);
	cv::Mat p_mat = hist.mul(_1_wh);
	double _entropy = 0.0;
	double pi = 0;
	for (int i = 0; i < 256; i++)
	{
		pi = (double)p_mat.at<float>(i);
		if (pi != 0.0)
		{
			_entropy += (pi * log2(pi));
		}
	}
	return -_entropy;
}

float FIMG::method_group::entropy_ab(cv::Mat& src1, cv::Mat& src2)
{
	//ͼ��������
	cv::Mat hist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
	int X, Y;
	for (int i = 0; i < src1.rows - 1; i++)
	{
		for (int j = 0; j < src1.cols - 1; j++)
		{
			X = src1.at<uchar>(i, j);
			Y = src1.at<uchar>(i, j);
			hist.at<uchar>(X, Y) += 1;
		}
	}
	cv::Mat histf;
	hist.convertTo(histf, CV_32F);
	double _1_wh = 1.0 / ((long long)src1.cols * (long long)src1.rows);
	cv::Mat p_mat = histf.mul(_1_wh);

	double _entropy = 0.0, pi = 0;
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			pi = (double)p_mat.at<float>(i, j);
			if (pi != 0.0)
			{
				_entropy += (pi * log2(pi));
			}
		}
	}
	return -_entropy;
}

float FIMG::method_group::entropy_2(cv::Mat& src)
{
	//ͼ���ά��
	cv::Mat kernel = (cv::Mat_<double>(3, 3) << 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 8, 0.0, 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 8);
	cv::Mat gray_filter;
	filter2D(src, gray_filter, -1, kernel);
	double _entropy = 0.0;
	_entropy = entropy_ab(src, gray_filter);
	return _entropy;

}

float FIMG::method_group::multi_info(cv::InputArray& src1, cv::InputArray& src2, cv::InputArray& fusedimg)
{
	cv::Mat img1 = src1.getMat();
	cv::Mat img2 = src2.getMat();
	cv::Mat img_f = fusedimg.getMat();

	float EN_1 = entropy_a(img1);
	float EN_2 = entropy_a(img2);
	float EN_12 = entropy_ab(img1, img2);
	if (!fusedimg.empty())
	{
		float EN_f = entropy_a(img_f);
		float EN_1f = entropy_ab(img1, img_f);
		float EN_2f = entropy_ab(img2, img_f);
		//return (EN_1f + EN_2f) / (EN_1 + EN_2);//�ںϻ���Ϣ��������ͼ�Ļ���Ϣ=EN1+EN2-EN12
		return EN_1 + EN_2 + 2 * EN_f - EN_1f - EN_2f;
	}
	else
	{
		return EN_1 + EN_2 - EN_12;
	}
}

float FIMG::method_group::mean_grad(cv::Mat& src)
{
	//ͼ��ƽ���ݶ�
	double sum_temp = 0;
	for (int i = 0; i < src.rows - 1; i++)
	{
		for (int j = 0; j < src.cols - 1; j++)
		{
			double temp1 = pow(src.at<uchar>(i, j) - src.at<uchar>(i, j + 1), 2);
			double temp2 = pow(src.at<uchar>(i, j) - src.at<uchar>(i + 1, j), 2);
			sum_temp += pow((temp1 + temp2) / 2, 0.5);
		}
	}
	return sum_temp / ((src.cols - 1) * (src.rows - 1));
}

float FIMG::method_group::calc_SSIM(cv::Mat& src, cv::Mat& fuse)
{
	//�ɷ�������м��㣬�˴�ֱ�Ӽ���

	cv::Mat u1, u2, sd1, sd2;
	meanStdDev(src, u1, sd1);
	meanStdDev(fuse, u2, sd2);
	double mean_1 = 0.0, mean_2 = 0.0, sigma_1 = 0.0, sigma_2 = 0.0, sigma_12 = 0.0;
	mean_1 = u1.at<double>(0, 0);
	mean_2 = u2.at<double>(0, 0);
	sigma_1 = sd1.at<double>(0, 0);
	sigma_2 = sd2.at<double>(0, 0);
	cv::Mat src_m = src - mean_1;
	cv::Mat fuse_m = fuse - mean_2;
	cv::Mat temp = src_m.mul(fuse_m);
	auto tttt = cv::mean(temp);
	sigma_12 = tttt[0];

	double C1 = 6.5025, C2 = 58.5225;
	double fenzi = (2 * mean_1 * mean_2 + C1) * (2 * sigma_1 * sigma_2 + C2);
	double fenmu = (mean_1 * mean_1 + mean_2 * mean_2 + C1) * (sigma_1 * sigma_1 + sigma_2 * sigma_2 + C2);
	return fenzi / fenmu;
}

float FIMG::method_group::calc_SF(cv::Mat& src)
{
	if (src.empty())
	{
		return -1;
	}
	float fImg_RF = 0.0, fImg_CF = 0.0;
	for (int i = 1; i < src.rows; i++)
	{
		for (int j = 1; j < src.cols; j++)
		{
			fImg_RF += pow((float)fabs(src.at<uchar>(i, j) - src.at<uchar>(i, j - 1)), 2);
			fImg_CF += pow((float)fabs(src.at<uchar>(i, j) - src.at<uchar>(i - 1, j)), 2);
		}
	}
	int W_H_1 = (src.rows - 1) * (src.cols);
	return pow((fImg_RF + fImg_CF) / W_H_1, 0.5);
}

float FIMG::method_group::calc_CC(cv::Mat& src1, cv::Mat& src2)
{
	if (src1.empty() || src2.empty())
	{
		return -1;
	}

	auto src1_mean = mean(src1);
	auto src2_mean = mean(src2);

	cv::Mat src1_temp = src1 - src1_mean[0];
	cv::Mat src2_temp = src2 - src2_mean[0];
	double dot_11 = 0.0, dot_12 = 0.0, dot_22 = 0.0;
	dot_11 = src1_temp.dot(src1_temp);
	dot_12 = src1_temp.dot(src2_temp);
	dot_22 = src2_temp.dot(src2_temp);

	return dot_12 / sqrt(dot_11 * dot_22);
}

void FIMG::method_group::hist_graph(cv::Mat hist)
{
	int scale = 2;
	int hist_height = 256;
	cv::Mat hist_img = cv::Mat::zeros(hist_height, 256 * scale, CV_8UC3); //����һ���ڵ׵�8λ��3ͨ��ͼ�񣬸�256����256*2
	double max_val;
	minMaxLoc(hist, 0, &max_val, 0, 0);//����ֱ��ͼ���������ֵ
	//�����صĸ������ϵ� ͼ������Χ��
	//����ֱ��ͼ�õ�������
	for (int i = 0; i < 256; i++)
	{
		float bin_val = hist.at<float>(i);   //����histԪ�أ�ע��hist����float���ͣ�
		int intensity = cvRound(bin_val * hist_height / max_val);  //���Ƹ߶�
		rectangle(hist_img, cv::Point(i * scale, hist_height - 1), cv::Point((i + 1) * scale - 1, hist_height - intensity), cv::Scalar(255, 255, 255));//����ֱ��ͼ
	}
	//imshow("hist_img", hist_img);
	//cv::waitKey();
}

cv::Mat FIMG::method_group::GuidedFilter(cv::Mat& input, cv::Mat& guidImg, int r, double eps)
{
	int wsize = 2 * r + 1;
	cv::Mat I, p;
	//��������ת��
	guidImg.convertTo(I, CV_64F, 1.0 / 255.0);
	input.convertTo(p, CV_64F, 1.0 / 255.0);


	//meanI=f_mean(I)
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//�����˲�
	//meanP=f_mean(P)
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//�����˲�

	//corrI=fmean(I.*I)
	cv::Mat mean_II;
	mean_II = I.mul(I);
	cv::boxFilter(mean_II, mean_II, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//�����˲�
	//corrIp=fmean(I.*p)
	cv::Mat mean_Ip;
	mean_Ip = I.mul(p);
	cv::boxFilter(mean_Ip, mean_Ip, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//�����˲�


	//varI=corrI-meanI.*meanI
	cv::Mat var_I, mean_mul_I;
	mean_mul_I = mean_I.mul(mean_I);
	//cv::subtract(mean_II, mean_mul_I, var_I);
	var_I = mean_II - mean_mul_I;

	//covIp=corrIp-meanI.*meanp
	cv::Mat cov_Ip;
	//cv::subtract(mean_Ip, mean_I.mul(mean_p), cov_Ip);
	cov_Ip = mean_Ip - mean_I.mul(mean_p);


	//a=conIp./(varI+eps)
	//b=meanp-a.*meanI
	cv::Mat a, b;
	//cv::divide(cov_Ip, (var_I + eps), a);
	//cv::subtract(mean_p, a.mul(mean_I), b);
	a = cov_Ip / (var_I + eps);
	b = mean_p - a.mul(mean_I);

	//mean_a=f_mean(a)
	//mean_b=f_mean(b)
	cv::Mat mean_a, mean_b;
	cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//�����˲�
	cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//�����˲�

	//q=meana.*I+meanb
	cv::Mat out;
	out = mean_a.mul(I) + mean_b;

	//��������ת��
	//I.convertTo(guidImg, CV_8U, 255);
	//p.convertTo(input, CV_8U, 255);
	out.convertTo(out, CV_8U, 255);

	return out;
}

int FIMG::method_group::Max_Entropy(cv::Mat& src, cv::Mat& dst, int thresh /*= 0*/, int p /*= 10*/)
{
	const int Grayscale = 256;
	int Graynum[Grayscale] = { 0 };
	int r = src.rows;
	int c = src.cols;
	for (int i = 0; i < r; ++i) {
		const uchar* ptr = src.ptr<uchar>(i);
		for (int j = 0; j < c; ++j) {
			if (ptr[j] == 0)				//�ų�����ɫ�����ص�
				continue;
			Graynum[ptr[j]]++;
		}
	}

	float probability = 0.0; //����
	float max_Entropy = 0.0; //�����
	int totalpix = r * c;
	for (int i = 0; i < Grayscale; ++i) {

		float HO = 0.0; //ǰ����
		float HB = 0.0; //������

		//����ǰ��������
		int frontpix = 0;
		for (int j = 0; j < i; ++j) {
			frontpix += Graynum[j];
		}
		//����ǰ����
		for (int j = 0; j < i; ++j) {
			if (Graynum[j] != 0) {
				probability = (float)Graynum[j] / frontpix;
				HO = HO + probability * log(1 / probability);
			}
		}

		//���㱳����
		for (int k = i; k < Grayscale; ++k) {
			if (Graynum[k] != 0) {
				probability = (float)Graynum[k] / (totalpix - frontpix);
				HB = HB + probability * log(1 / probability);
			}
		}

		//���������
		if (HO + HB > max_Entropy) {
			max_Entropy = HO + HB;
			thresh = i + p;
		}
	}

	//��ֵ����
	src.copyTo(dst);
	for (int i = 0; i < r; ++i) {
		uchar* ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < c; ++j) {
			if (ptr[j] > thresh)
				ptr[j] = 255;
			else
				ptr[j] = 0;
		}
	}
	return thresh;
}

int FIMG::method_group::Adjust_gamma(cv::Mat& src, cv::Mat& dst, float gamma /*= 1.0*/)
{
	if (gamma == 1)
	{
		auto mean_val = mean(src);
		gamma = log10(0.5) / log10(mean_val[0] / 255);
	}
	float table[256];
	for (int i = 0; i < 256; i++)
	{
		table[i] = std::pow(i / 255.0, gamma) * 255;
	}

	dst = cv::Mat::zeros(src.size(), src.type());
	if (src.channels() == 3)
	{
		for (int row = 0; row < src.rows - 1; row++)
		{
			for (int col = 0; col < src.cols - 1; col++)
			{
				dst.at<cv::Vec3b>(row, col)[0] = (uchar)table[src.at<cv::Vec3b>(row, col)[0]];
				dst.at<cv::Vec3b>(row, col)[1] = (uchar)table[src.at<cv::Vec3b>(row, col)[1]];
				dst.at<cv::Vec3b>(row, col)[2] = (uchar)table[src.at<cv::Vec3b>(row, col)[2]];
			}
		}
	}
	else
	{
		for (int row = 0; row < src.rows - 1; row++)
		{
			for (int col = 0; col < src.cols - 1; col++)
			{
				dst.at<uchar>(row, col) = (uchar)table[src.at<uchar>(row, col)];//��������ټ�����	
			}
		}
	}

	return true;
}

int FIMG::method_group::Adjust_contrast_brightness(cv::Mat& src, cv::Mat& dst, double alpha, double beta)
{
	dst = cv::Mat::zeros(src.size(), src.type());
	if (src.channels() == 3)
	{
		for (int row = 0; row < src.rows - 1; row++) {
			for (int col = 0; col < src.cols - 1; col++) {
				// ��ȡ����
				int b = src.at<cv::Vec3b>(row, col)[0];
				int g = src.at<cv::Vec3b>(row, col)[1];
				int r = src.at<cv::Vec3b>(row, col)[2];
				// �������ȺͶԱȶ�
				dst.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(alpha * b + beta);
				dst.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(alpha * g + beta);
				dst.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(alpha * r + beta);
			}
		}
	}
	else
	{
		for (int row = 0; row < src.rows - 1; row++) {
			for (int col = 0; col < src.cols - 1; col++) {
				// ��ȡ����
				int scalar = src.at<uchar>(row, col);

				// �������ȺͶԱȶ�
				dst.at<uchar>(row, col) = cv::saturate_cast<uchar>(alpha * scalar + beta);

			}
		}
	}
	return true;
}

int FIMG::method_group::Affine_trans_base_matrix(cv::Mat& src, cv::Point2f& trans_center, cv::Mat& dst, affine_trans_params& param, bool fullDisplayImg)
{
	if (src.empty()) { return false; }

	double Angle = param.rotate_angle * CV_PI / 180.0;
	double alpha = param.x_scale * cos(Angle);
	double beta = param.y_scale * sin(Angle);
	cv::Mat mat_trans_temp = (cv::Mat_<double>(2, 3) << alpha, beta, ((1.0 - alpha) * trans_center.x - beta * trans_center.y + param.x_offset),
		-beta, alpha, (beta * trans_center.x + (1.0 - alpha) * trans_center.y + param.y_offset));
	int dst_rows = 0, dst_cols = 0;
	if (fullDisplayImg)
	{
		dst_rows = round(fabs(param.y_scale * src.rows * cos(Angle)) + fabs(param.x_scale * src.cols * sin(Angle)));//�پ�����ת��ͼ��߶�
		dst_cols = round(fabs(param.x_scale * src.cols * cos(Angle)) + fabs(param.y_scale * src.rows * sin(Angle)));//�پ�����ת��ͼ����
		mat_trans_temp.at<double>(0, 2) += ((dst_cols - src.cols) / 2 - param.x_offset);//ƽ����ʾȫͼ
		mat_trans_temp.at<double>(1, 2) += ((dst_rows - src.rows) / 2 - param.y_offset);
	}
	warpAffine(src, dst, mat_trans_temp, cv::Size(dst_cols, dst_rows));

	return true;
}

void FIMG::method_group::calc_clache_hist(cv::Mat& src, cv::Mat& dst, double dValue /*= 40.0*/, cv::Size img_block /*= cv::Size(8, 8)*/)
{
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(dValue, img_block);
	clahe->apply(src, dst);
}

int FIMG::method_group::Alpha_Beta_Image_fuse(cv::Mat& src1, cv::Mat& src2, cv::Mat& dst, double dAlpha, double gamma /*= 0*/)
{
	if (src1.size() != src2.size()) {
		//printf("resize start!\n");
		//resize(imsrc2, imsrc2_scaler, Size(imsrc1.cols, imsrc1.rows), 0, 0, INTER_LINEAR);
		cv::Mat src2_1;
		resize(src2, src2_1, src1.size(), 0, 0, cv::INTER_LINEAR);    //����2
		addWeighted(src1, dAlpha, src2_1, 1 - dAlpha, gamma, dst);
	}
	else {
		addWeighted(src1, dAlpha, src2, 1 - dAlpha, gamma, dst);
	}
	return true;
}

int FIMG::method_group::GrayStretch(cv::Mat& src, cv::Mat& dst, double dmin_s /*= 0.0*/, double dmax_s /*= 255.0*/)
{
	if (src.empty()) { return false; }
	cv::Mat gray;
	if (src.channels() == 3)
	{
		cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	}
	else
	{
		src.copyTo(gray);
	}
	dst = cv::Mat(gray.size(), CV_8UC1);
	double min_Val, max_Val;
	cv::Point min_p, max_p;
	minMaxLoc(gray, &min_Val, &max_Val, &min_p, &max_p);
	double k1 = dmin_s / min_Val;
	double k2 = (dmax_s - dmin_s) / (max_Val - min_Val);
	double k3 = (255.0 - dmax_s) / (255 - max_Val);
	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			if (gray.at<uchar>(i, j) < min_Val)
			{
				dst.at<uchar>(i, j) = (uchar)(gray.at<uchar>(i, j) * k1);
			}
			else if (gray.at<uchar>(i, j) >= min_Val && gray.at<uchar>(i, j) <= max_Val)
			{
				dst.at<uchar>(i, j) = (uchar)((gray.at<uchar>(i, j) - min_Val) * k2 + dmin_s);
			}
			else if (gray.at<uchar>(i, j) > max_Val)
			{
				dst.at<uchar>(i, j) = (uchar)((gray.at<uchar>(i, j) - max_Val) * k3 + dmax_s);
			}
		}
	}
}

void FIMG::method_group::SalientRegionDetectionBasedonLC(cv::Mat& src, cv::Mat& dst)
{
	int row = src.rows, col = src.cols;

	int val;
	int HistGram[256] = { 0 };
	cv::Mat gray = cv::Mat(src.size(), CV_8UC1);
	dst = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::Point3_<uchar>* p;
	//ͳ�ƻҶȼ�����
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			p = src.ptr<cv::Point3_<uchar> >(i, j);
			val = (p->x + (p->y) * 2 + p->z) / 4;
			HistGram[val]++;
			gray.at<uchar>(i, j) = val;
		}
	}



	int Dist[256] = { 0 };
	int Y, X;
	int max_gray = 0;
	int min_gray = 1 << 28;

	for (Y = 0; Y < 256; Y++)
	{
		for (X = 0; X < 256; X++)
			Dist[Y] += abs(Y - X) * HistGram[X];//���Ĺ�ʽ��9�����Ҷȵľ���ֻ�о���ֵ��������ʵ�����Ż��ٶȣ�������������û��Ҫ��  
		max_gray = cv::max(max_gray, Dist[Y]);
		min_gray = cv::min(min_gray, Dist[Y]);
	}

	int max_min = max_gray - min_gray;

	for (Y = 0; Y < row; Y++)
	{
		for (X = 0; X < col; X++)
		{
			dst.at<uchar>(Y, X) = (Dist[gray.at<uchar>(Y, X)] - min_gray) * 255 / max_min;//����ȫͼÿ�����ص������� 
			//Sal.at<uchar>(Y,X) = (Dist[gray.at<uchar>(Y, X)])*255/(max_gray);//����ȫͼÿ�����ص�������  
		}
	}
}

int FIMG::method_group::Get_Infrared_target_region(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat gray, gamma, gray_median;
	if (src.empty()) { return false; }
	else
	{
		if (src.channels() > 1)
		{
			cvtColor(src, gray, cv::COLOR_BGR2GRAY);
		}
		else
		{
			src.copyTo(gray);
		}
	}
	if (mean(gray)[0] > 100)
	{
		Adjust_gamma(gray, gamma, 1.8);
	}
	else
	{
		gray.copyTo(gamma);
	}
	//medianBlur(gamma, gray_median, 3);

	Max_Entropy(gamma, dst, 0, 7);

	return true;
}

void FIMG::method_group::RegionGrow(cv::Mat src, cv::Mat& matDst)
{
	int DIR[8][2] = { { -1, -1 },{ 0, -1 },{ 1, -1 },{ 1, 0 },{ 1, 1 },{ 0, 1 },{ -1, 1 },{ -1, 0 } };//��������˳������//loop
	cv::Point2i pt = cv::Point(0, src.rows);//��ʼλ�ø��ݣ����������ҳ� ����loop����ʹ����ʱ�临�Ӹ����
	std::vector<cv::Point2i> vcGrowPt;						//������ջ
	vcGrowPt.push_back(pt);							//��������ѹ��ջ��

	double min_g, max_g;
	minMaxLoc(src, &min_g, &max_g);
	int iThre = (max_g + 4 * min_g) / 5;						//��������ҶȲ���ֵ


	int nSrcValue = 0;								//�������Ҷ�ֵ����ʼ�Ҷȶ�Ϊ��ֵ�Ҷ�
	int nCurValue = 0;								//��ǰ������Ҷ�ֵ

	matDst = cv::Mat(src.size(), CV_8UC1, cv::Scalar(255));	//����һ���հ��������Ϊ��ɫ												
	matDst.at<uchar>(pt.y, pt.x) = 0;				//���������

	//nSrcValue = src.at<uchar>(pt.y, pt.x);		  //��¼������ĻҶ�ֵ������ͼ��Ϊ��׼������ֵΪ��׼

	nSrcValue = mean(src)[0];						//��ʵ�Ҷȶ�λ�Ҷȿ������1/3��

	cv::Point2i ptGrowing;							 //��������λ��	
	while (!vcGrowPt.empty())						//����ջ��Ϊ��������
	{
		pt = vcGrowPt.back();						//ȡ��һ��������
		vcGrowPt.pop_back();
		//�ֱ�԰˸������ϵĵ��������
		for (int i = 0; i < 8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//����Ƿ��Ǳ�Ե��//loop
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;
			int nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//��ǰ��������ĻҶ�ֵ
			if (nGrowLable == 255)					//�����ǵ㻹û�б�����
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				//if (abs(nSrcValue - nCurValue) < th)					//����ֵ��Χ��������
				if (nCurValue - nSrcValue < iThre)	//�޸�Ϊ������ֵ��
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 0;		//���Ϊ��ɫ
					vcGrowPt.push_back(ptGrowing);					//����һ��������ѹ��ջ��
				}
			}
		}
	}
	return;
}

string FIMG::method_group::getCurFilePath()
{
	char buffer[MAX_PATH];
	auto x = _getcwd(buffer, MAX_PATH);
	string path = buffer;
	replace(path.begin(), path.end(), '\\', '/');
	return path;
}

//�ļ����� �б��ȡ����������Windows,Linux ��ע��
void FIMG::method_group::listFiles(string dir, vector<string>& files, string str_img_type)
{
	string filename;
	filename.assign(dir).append("*." + str_img_type);//ƥ���ļ�����
	intptr_t handle;
	_finddata_t findData;//��ȡ�ļ���
	handle = _findfirst(filename.c_str(), &findData);
	if (handle == -1)        // ����Ƿ�ɹ�
		return;
	do
	{
		string xxx = dir;
		xxx.append(findData.name);
		files.push_back(xxx);
		//files.push_back(filename);
	} while (_findnext(handle, &findData) == 0);
	_findclose(handle);    // �ر��������
}
