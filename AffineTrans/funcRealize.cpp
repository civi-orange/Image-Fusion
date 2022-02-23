#include "funcHead.h"

//windows ��ȡָ���ļ������ļ�����  ���ִ˾��� ����Ԥ�����ļ���ʼ����#define _CRT_SECURE_NO_DEPRECATE
void listFiles(string dir, vector<string>& files, string str_img_type)
{
	string filename;
	filename.assign(dir).append("*."+ str_img_type);//ƥ���ļ�����
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


void CalcCorners(const cv::Mat& H, const cv::Mat& src, four_corners_t& corners)//��H*SRC����ͼ�ǵ�����
{
	double v2[] = { 0, 0, 1 };//���Ͻ�
	double v1[3];//�任�������ֵ
	cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);  //������
	cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1);  //������

	V1 = H * V2;
	//���Ͻ�(0,0,1)
	//cout << "V2: " << endl << V2 << endl;
	//cout << "V1: " << endl << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//���½�(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //������
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//���Ͻ�(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //������
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//���½�(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //������
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

	return;
}


void RegionGrow(cv::Mat src, cv::Mat& matDst)
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

int Get_Infrared_target_region(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat gray, gamma, gray_median, gray_mid, rg_mat, gray_open;	
	if (src.empty()){return false;}
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
	resize(gamma, gray_mid, cv::Size(src.cols / 2, src.rows / 2), 0, 0, 3);
	//RegionGrow(gray_mid, rg_mat);
	Max_Entropy(gray_mid, rg_mat, 0, 7);
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//morphologyEx(rg_mat, gray_open, cv::MORPH_DILATE, element);
	resize(rg_mat, dst, cv::Size(src.cols, src.rows), 0, 0, 3);

	//imshow("gamma", gamma);
	//imshow("small", rg_mat);
	//imshow("gray_open", gray_open);
	//imshow("big", dst);
	return true;
}


std::vector<cv::Mat> Gauss_Pyr(cv::Mat& input, std::vector<cv::Mat>& Img_pyr, int level)
{
	Img_pyr.push_back(input);

	cv::Mat dst;
	for (int i = 0; i < level; i++)
	{
		pyrDown(input, dst, cv::Size(input.cols / 2, input.rows / 2));
		Img_pyr.push_back(dst);
		input = dst;
	}
	return Img_pyr;
}

std::vector<cv::Mat> Laplace_Pyr(std::vector<cv::Mat>& Img_Gaussian_pyr, std::vector<cv::Mat>& Img_Laplacian_pyr, int level)
{
	cv::Mat img_sub, img_up, up, img_lp;

	for (int i = 0; i < level; i++)
	{
		img_sub = Img_Gaussian_pyr[i];
		img_up = Img_Gaussian_pyr[i + 1];
		pyrUp(img_up, up, cv::Size(img_up.cols * 2, img_up.rows * 2));//�����ϲ��� 
		img_lp = img_sub - up;
		Img_Laplacian_pyr.push_back(img_lp);
	}

	return Img_Laplacian_pyr;
}

std::vector<cv::Mat> Fusion_Laplace_Pyr(std::vector<cv::Mat>& Img_front_lp, std::vector<cv::Mat>& Img_back_lp, std::vector<cv::Mat>& mask_gau, std::vector<cv::Mat>& blend_lp)
{

	int level = Img_front_lp.size();

	for (int i = 0; i < level; i++)//0Ϊ��ͼ���������ײ�                                        
	{
		cv::Mat A = Img_front_lp[i].mul(mask_gau[i]); //Alpha��Attention:��Data types must be unified 
		cv::Mat antiMask;
		if (mask_gau[0].channels() == 3) {
			antiMask = cv::Scalar(1.0, 1.0, 1.0) - mask_gau[i];//��ɫͼ
		}
		else {
			antiMask = cv::Scalar(1.0) - mask_gau[i];//�Ҷ�ͼ
		}
		cv::Mat B = Img_back_lp[i].mul(antiMask); //Beta
		cv::Mat blendedLevel = A + B;
		blend_lp.push_back(blendedLevel);
	}
	return blend_lp;
}

int Laplace_pyramid_fusion(cv::Mat& frame_back, cv::Mat& frame_front, double dAlpha, int pyr_level, cv::Mat& output)
{
	//�ںϲ���0
	cv::Mat mask = cv::Mat::zeros(frame_front.size(), CV_32FC1);
	mask(cv::Range::all(), cv::Range::all()) = dAlpha;//��߶������ں�
	if (frame_front.channels() == 3) { cvtColor(mask, mask, cv::COLOR_GRAY2BGR); }
	std::vector<cv::Mat> mask_Pyr;
	mask_Pyr = Gauss_Pyr(mask, mask_Pyr, pyr_level);

	cv::Mat img_front, img_back;
	frame_front.convertTo(img_front, CV_32F);
	frame_back.convertTo(img_back, CV_32F);

	std::vector<cv::Mat> Gau_Pyr_back, lp_Pyr_back, Gau_Pyr_front, lp_Pyr_front, out_temp;
	Gauss_Pyr(img_back, Gau_Pyr_back, pyr_level);
	Gauss_Pyr(img_front, Gau_Pyr_front, pyr_level);
	Laplace_Pyr(Gau_Pyr_back, lp_Pyr_back, pyr_level);
	Laplace_Pyr(Gau_Pyr_front, lp_Pyr_front, pyr_level);
	Fusion_Laplace_Pyr(lp_Pyr_back, lp_Pyr_front, mask_Pyr, out_temp);

	int level = out_temp.size();

	cv::Mat img_up;

	cv::Mat img_start = Gau_Pyr_front.back().mul(mask_Pyr.back()) + ((Gau_Pyr_back.back()).mul(cv::Scalar(1.0) - mask_Pyr.back()));//�Ҷ�ͼ

	for (int i = 0; i < level; i++)
	{
		pyrUp(img_start, img_up, cv::Size(img_start.cols * 2, img_start.rows * 2));
		output = out_temp[level - i - 1] + img_up;
		img_start = output;
	}
	output.convertTo(output, CV_8UC1);
	return true;
}

int ImageMatch(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo, int detectorType)
{
	switch (detectorType)
	{
	case METHOD_SURF_DETECTOR:
		surf_Detector(img_back, img_front, homo);
		break;
	case METHOD_SIFT_DETECTOR:
		sift_Detector(img_back, img_front, homo);
		break;
	case METHOD_FAST_DETECTOR:
		fast_Detector(img_back, img_front, homo);
		break;
	default:
		break;
	}
	
	return true;
}

int fast_Detector(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo)
{
	//��ȡ������    
	cv::Ptr<cv::FastFeatureDetector> Detector = cv::FastFeatureDetector::create(50);  //��ֵ 
	vector<cv::KeyPoint> keyPoint_front, keyPoint_back;
	Detector->detect(img_front, keyPoint_front);
	Detector->detect(img_back, keyPoint_back);

	//������������Ϊ�±ߵ�������ƥ����׼��    
	cv::Ptr<cv::SiftDescriptorExtractor> SiftDescriptor = cv::SiftDescriptorExtractor::create();
	cv::Mat imageDesc_front, imageDesc_back;
	SiftDescriptor->compute(img_front, keyPoint_front, imageDesc_front);
	SiftDescriptor->compute(img_back, keyPoint_back, imageDesc_back);

	//BruteForceMatcher 
	//cv::Ptr<cv::BFMatcher> bfmatcher = cv::BFMatcher::create(cv::NORM_L2, true);
	cv::BFMatcher bfmatcher;
	vector<vector<cv::DMatch> > matchePoints;
	vector<cv::DMatch> GoodMatchePoints;

	vector<cv::Mat> train_desc_fast(1, img_front);//����1��Ԫ�ض�ΪimageDesc1
	bfmatcher.add(train_desc_fast);
	bfmatcher.train();
	bfmatcher.knnMatch(imageDesc_back, matchePoints, 2);
	//cout << "total match points: " << matchePoints.size() << endl;

	// Lowe's algorithm,��ȡ����ƥ���
	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (matchePoints[i][0].distance < 0.6 * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}

	cv::Mat first_match;
	drawMatches(img_back, keyPoint_back, img_front, keyPoint_front, GoodMatchePoints, first_match);
	cv::imshow("result_sift", first_match);

	//ͼ��ƥ��
	vector<cv::Point2f> imagePoints_front, imagePoints_back;
	for (int i = 0; i < GoodMatchePoints.size(); i++)
	{
		imagePoints_back.push_back(keyPoint_back[GoodMatchePoints[i].queryIdx].pt);//queryidxΪͼ2
		imagePoints_front.push_back(keyPoint_front[GoodMatchePoints[i].trainIdx].pt);
	}

	//��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
	homo = findHomography(imagePoints_front, imagePoints_back, cv::RANSAC);  // src = h * dst
	////Ҳ����ʹ��getPerspectiveTransform�������͸�ӱ任���󣬲���Ҫ��ֻ����4���㣬Ч���Բ�  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "�任����Ϊ��\n" << homo << endl << endl; //���ӳ�����

	four_corners_t corners;
	//������׼ͼ���ĸ���������
	CalcCorners(homo, img_front, corners);//��frame_front�Ľǵ�����ͨ��homoӳ�䵽frame_back���ĸ��ǵ�

	//ͼ����׼  
	cv::Mat imageTransform1, imageTransform2;
	warpPerspective(img_front, imageTransform1, homo, cv::Size(MAX(corners.right_top.x, corners.right_bottom.x), img_back.rows));//��frame_1��frame_2����ӳ�䣬��frame_2��rowsΪ��׼
	//warpPerspective(frame_1, imageTransform1, homo, cv::Size(MAX(corners.right_top.x, corners.right_bottom.x), MAX(corners.right_top.y, corners.right_bottom.y)));
	//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform1);


	//����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
	int dst_width = imageTransform1.cols;  //ȡ���ҵ�ĳ���Ϊƴ��ͼ�ĳ���
	int dst_height = img_back.rows;
	cv::Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	imageTransform1.copyTo(dst(cv::Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	img_back.copyTo(dst(cv::Rect(0, 0, img_back.cols, img_back.rows)));
	imshow("b_dst", dst);
	cv::waitKey();

	OptimizeSeam(img_back, imageTransform1, corners, dst);
	imshow("dst", dst);


	return true;
}

int surf_Detector(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo)
{
	return true;
}

int sift_Detector(cv::Mat& img_back, cv::Mat& img_front, cv::Mat& homo)
{
	return true;
}

//�Ż���ͼ�����Ӵ���ʹ��ƴ����Ȼ //�ص�������ALPHA�ں�
void OptimizeSeam(cv::Mat& img1, cv::Mat& trans, four_corners_t& corners, cv::Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//��ʼλ�ã����ص��������߽�  
	double processWidth = MIN(img1.cols, trans.cols) - start;//�ص�����Ŀ��  ��ȡimg1��ԭ�򣿣������tran.cols>img.cols�޿ɺ�ǣ����tran.cols<img.cols
	int rows = dst.rows;
	int cols = img1.cols; //ע�⣬������*ͨ����
	double alpha = 1;//img1�����ص�Ȩ��
	if (img1.channels() == 3)
	{
		for (int i = 0; i < rows; i++)
		{
			uchar* p = img1.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ
			uchar* t = trans.ptr<uchar>(i);
			uchar* d = dst.ptr<uchar>(i);
			for (int j = start; j < cols; j++)
			{
				//�������ͼ��trans�������صĺڵ㣬����ȫ����img1�е�����
				if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
				{
					alpha = 1;
				}
				else
				{
					//img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ�������ȣ�ʵ��֤�������ַ���ȷʵ��  
					alpha = (processWidth - (j - start)) / processWidth;
				}
				d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
				d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
				d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

			}
		}
	}
	else if (img1.channels() == 1)
	{
		for (int i = 0; i < rows; i++)
		{
			uchar* p = img1.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ
			uchar* t = trans.ptr<uchar>(i);
			uchar* d = dst.ptr<uchar>(i);
			for (int j = start; j < cols; j++)
			{
				//�������ͼ��trans�������صĺڵ㣬����ȫ����img1�е�����
				if (t[j] == 0)
				{
					alpha = 1;
				}
				else
				{
					//img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ�������ȣ�ʵ��֤�������ַ���ȷʵ��  
					alpha = (processWidth - (j - start)) / processWidth;
				}
				d[j] = p[j] * alpha + t[j] * (1 - alpha);
			}
		}
	}

}


void CountHistNum(cv::Mat& src, std::vector<int>& out)
{
	cv::Mat dst;
	int v[256] = { 0 };

	int row = src.rows, col = src.cols;
	
	if (src.channels()>1)
	{
		cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	}
	
	//ͳ�ƻҶȼ�����
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			int val = dst.at<uchar>(i,j);
			v[val]++;
		}
	}
	out = vector<int>(v, v + 256);
	return;
}

//2022/01/18 ��ӹ��̱��ļ�û���������
void SalientRegionDetectionBasedonLC(cv::Mat& src, cv::Mat& dst) 
{
	
	int row = src.rows, col = src.cols; 
	
	//cv::Mat gray;
	//if (src.channels()>1)
	//{
	//	cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	//}
	//else
	//{
	//	src.copyTo(gray);
	//}
	//dst = cv::Mat::zeros(src.size(), CV_8UC1);
	//std::vector<int> HistGram;
	//CountHistNum(src, HistGram);

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



	int Dist[256] = {0};
	int Y, X;
	int max_gray = 0;
	int min_gray = 1 << 28;	

	for (Y = 0; Y < 256; Y++)
	{
		for (X = 0; X < 256; X++)
			Dist[Y] += abs(Y - X) * HistGram[X];//���Ĺ�ʽ��9�����Ҷȵľ���ֻ�о���ֵ��������ʵ�����Ż��ٶȣ�������������û��Ҫ��  
		max_gray = max(max_gray, Dist[Y]);
		min_gray = min(min_gray, Dist[Y]);
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

//�Ҷ�����
int GrayStretch(cv::Mat& src, cv::Mat& dst, double dmin_s, double dmax_s)
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
	for (int i=0; i< gray.rows; i++)
	{
		for (int j=0; j< gray.cols; j++)
		{
			if (gray.at<uchar>(i,j) < min_Val)
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


int Alpha_Beta_Image_fuse(cv::Mat& src1, cv::Mat& src2, cv::Mat& dst, double dAlpha, double gamma)
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

//����任
int Affine_trans_base_matrix(cv::Mat& src, cv::Point2f& trans_center, affine_trans_params& param, cv::Mat& dst, bool fullDisplayImg)
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


int Adjust_contrast_brightness(cv::Mat& src, cv::Mat& dst, double alpha, double beta)
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

void calc_clache_hist(cv::Mat& src, cv::Mat& dst, double dValue, cv::Size img_block)
{
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(dValue, img_block);
	clahe->apply(src, dst);
}

int Adjust_gamma(cv::Mat& src, cv::Mat& dst, float gamma)
{	

	if (gamma == 1)
	{
		auto mean_val = mean(src);
		gamma = log10(0.5) / log10(mean_val[0] / 255);
	}
	float table[256];
	for (int i = 0; i < 256; i++)
	{
		table[i] = std::pow(i / 255.0, gamma)* 255;
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

//ͼ����
float entropy_a(cv::Mat& src)
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

//ͼ��������
float entropy_ab(cv::Mat& src1, cv::Mat& src2)
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

//ͼ���ά��
float entropy_2(cv::Mat& src)
{
	//ͼ���ά��
	cv::Mat kernel = (cv::Mat_<double>(3, 3) << 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 8, 0.0, 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 8);
	cv::Mat gray_filter;
	filter2D(src, gray_filter, -1, kernel);
	double _entropy = 0.0;
	_entropy = entropy_ab(src, gray_filter);
	return _entropy;
}

//ͼ����ϢMI
float multi_info(cv::InputArray& src1, cv::InputArray& src2, cv::InputArray& fusedimg)
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

//ͼ��ƽ���ݶ�
float mean_grad(cv::Mat& src)
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

//SSIM�ṹ�����ƶ�
float calc_SSIM(cv::Mat& src, cv::Mat& fuse)
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
	double fenzi = (2 * mean_1 * mean_2 + C1) * (2 * sigma_1*sigma_2 + C2);
	double fenmu = (mean_1 * mean_1 + mean_2 * mean_2 + C1) * (sigma_1*sigma_1 + sigma_2*sigma_2 + C2);
	return fenzi / fenmu;

}

//�ռ�Ƶ��
float calc_SF(cv::Mat& src)
{
	if (src.empty())
	{
		return -1;
	}
	float fImg_RF = 0.0, fImg_CF = 0.0;
	for (int i=1; i<src.rows; i++)
	{
		for (int j=1; j<src.cols; j++)
		{
			fImg_RF += pow((float)fabs(src.at<uchar>(i, j) - src.at<uchar>(i, j - 1)), 2);
			fImg_CF += pow((float)fabs(src.at<uchar>(i, j) - src.at<uchar>(i - 1, j)), 2);
		}
	}
	int W_H_1 = (src.rows - 1) * (src.cols);
	return pow((fImg_RF + fImg_CF) / W_H_1, 0.5);
}

//���ϵ��
float calc_CC(cv::Mat& src1, cv::Mat& src2)
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

	return dot_12 / sqrt(dot_11*dot_22);
}

//��Ե��Ϣ������
float calc_QAB_F(cv::Mat& src1, cv::Mat& src2, cv::Mat& fuse)
{

	return -1;
}

//��ͼ��ֱ��ͼ
void hist_graph(cv::Mat hist)
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
	imshow("hist_img", hist_img);
	cv::waitKey();
}



int image_fusion_evalution(cv::Mat frame_IR, cv::Mat frame_vis, cv::Mat frame_fuse, fusion_image_index& output)
{
	if (frame_IR.empty() || frame_vis.empty() || frame_fuse.empty())
	{
		return false;
	}
	//EN
	output.fusion_EN = entropy_a(frame_fuse);

	//MI
	output.fusion_MI = multi_info(frame_IR, frame_vis, frame_fuse);
	//mean grad
	output.fusion_meanGrad = mean_grad(frame_fuse);
	//SSIM
	output.fusion_SSIM_irf = calc_SSIM(frame_IR, frame_fuse);
	output.fusion_SSIM_visf = calc_SSIM(frame_vis, frame_fuse);
	//SF
	output.fusion_SF = calc_SF(frame_fuse);
	//CC
	output.fusion_CC_irf = calc_CC(frame_IR, frame_fuse);
	output.fusion_CC_visf = calc_CC(frame_vis, frame_fuse);
	//SD
	cv::Mat mean_val, sd_val;
	meanStdDev(frame_fuse, mean_val, sd_val);
	output.fusion_SD = sd_val.at<double>(0, 0);
	//Qab/f
	output.fusion_Qab_f = -1;//δʵ��
	//VIF
	output.fusion_VIF = -1;//δʵ��

	return true;
}


cv::Mat GuidedFilter(cv::Mat& input, cv::Mat& guidImg, int r, double eps)
{	
	int wsize = 2 * r + 1;	
	cv::Mat I, p;
	//��������ת��
	guidImg.convertTo(I, CV_64F, 1.0 / 255.0);
	input.convertTo(p, CV_64F, 1.0 / 255.0);


	//meanI=fmean(I)
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//�����˲�
	//meanP=fmean(P)
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

	//meana=fmean(a)
	//meanb=fmean(b)
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

//ͼ�� �Ա���ʾ
cv::Mat All_Images_inWindow(vector<cv::Mat> vct_img)
{

	int w = vct_img[0].cols;
	int h = vct_img[0].rows;
	cv::Mat result = cv::Mat::zeros(cv::Size(2 * w, 2 * h), vct_img[0].type());
	cv::Rect box(0, 0, w, h);
	for (int i = 0; i < 4; i++) {
		int row = i / 2;
		int col = i % 2;
		box.x = w * col;
		box.y = h * row;
		vct_img[i].copyTo(result(box));
	}
	return result;
}

string getCurFilePath()
{
	char buffer[MAX_PATH];
	auto x = _getcwd(buffer, MAX_PATH);
	string path = buffer;
	replace(path.begin(), path.end(), '\\', '/');
	return path;
}








//����طָ��㷨 ��ֵ�ָp�ǵ��������ֶ�
int Max_Entropy(cv::Mat& src, cv::Mat& dst, int thresh, int p) 
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


///ȥ���㷨�� ʵ��ʾ��
//int main()
//{
//	cv::Mat src = cv::imread("C:/Users/18301/Desktop/1.png");
//	cv::Mat dst;
//	cvtColor(src, dst, cv::COLOR_BGR2GRAY);
//	cv::Mat dark_channel_mat = dark_channel(src);//������ǰ�ͨ��ͼ��
//	int A = calculate_A(src, dark_channel_mat);
//	cv::Mat tx = calculate_tx(src, A, dark_channel_mat);
//	cv::Mat tx_ = guidedfilter(dst, tx, 30, 0.001);//�����˲����tx��dstΪ����ͼ��
//	cv::Mat haze_removal_image;
//	haze_removal_image = haze_removal_img(src, A, tx_);
//	cv::namedWindow("ȥ����ͼ��", 0);
//	cv::namedWindow("ԭʼͼ��", 0);
//	imshow("ԭʼͼ��", src);
//	imshow("ȥ����ͼ��", haze_removal_image);
//	cv::waitKey(0);
//	return 0;
//}

//*********������������������������������ȥ���㷨����Ҫ���� ��������������������������������*********//
//�����˲��������Ż�t(x)����Ե�ͨ��
cv::Mat guidedfilter(cv::Mat& srcImage, cv::Mat& srcClone, int r, double eps)
{
	//ת��Դͼ����Ϣ
	srcImage.convertTo(srcImage, CV_32FC1, 1 / 255.0);
	srcClone.convertTo(srcClone, CV_32FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	cv::Mat boxResult;
	//����һ�������ֵ
	boxFilter(cv::Mat::ones(nRows, nCols, srcImage.type()), boxResult, CV_32FC1, cv::Size(r, r));
	//���ɵ����ֵmean_I
	cv::Mat mean_I;
	boxFilter(srcImage, mean_I, CV_32FC1, cv::Size(r, r));
	//����ԭʼ��ֵmean_p
	cv::Mat mean_p;
	boxFilter(srcClone, mean_p, CV_32FC1, cv::Size(r, r));
	//���ɻ���ؾ�ֵmean_Ip
	cv::Mat mean_Ip;
	boxFilter(srcImage.mul(srcClone), mean_Ip, CV_32FC1, cv::Size(r, r));
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	//��������ؾ�ֵmean_II
	cv::Mat mean_II;
	//Ӧ�ú��˲���������ص�ֵ
	boxFilter(srcImage.mul(srcImage), mean_II, CV_32FC1, cv::Size(r, r));
	//��������������ϵ��
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);
	cv::Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
	//���������������ϵ��a,b
	cv::Mat a = cov_Ip / (var_I + eps);
	cv::Mat b = mean_p - a.mul(mean_I);
	//������: ����ϵ�� a/b �ľ�ֵ
	cv::Mat mean_a;
	boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
	mean_a = mean_a / boxResult;
	cv::Mat mean_b;
	boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));
	mean_b = mean_b / boxResult;
	//�����壺�����������
	cv::Mat resultMat = mean_a.mul(srcImage) + mean_b;
	return resultMat;
}

//���㰵ͨ��ͼ����������ͨ����ɫͼ��
cv::Mat dark_channel(cv::Mat src)
{
	int border = 7;
	std::vector<cv::Mat> rgbChannels(3);
	cv::Mat min_mat(src.size(), CV_8UC1, cv::Scalar(0)), min_mat_expansion;
	cv::Mat dark_channel_mat(src.size(), CV_8UC1, cv::Scalar(0));
	split(src, rgbChannels);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int min_val = 0;
			int val_1, val_2, val_3;
			val_1 = rgbChannels[0].at<uchar>(i, j);
			val_2 = rgbChannels[1].at<uchar>(i, j);
			val_3 = rgbChannels[2].at<uchar>(i, j);

			min_val = std::min(val_1, val_2);
			min_val = std::min(min_val, val_3);
			min_mat.at<uchar>(i, j) = min_val;

		}
	}
	copyMakeBorder(min_mat, min_mat_expansion, border, border, border, border, cv::BORDER_REPLICATE);

	for (int m = border; m < min_mat_expansion.rows - border; m++)
	{
		for (int n = border; n < min_mat_expansion.cols - border; n++)
		{
			cv::Mat imageROI;
			int min_num = 256;
			imageROI = min_mat_expansion(cv::Rect(n - border, m - border, 2 * border + 1, 2 * border + 1));
			for (int i = 0; i < imageROI.rows; i++)
			{
				for (int j = 0; j < imageROI.cols; j++)
				{
					int val_roi = imageROI.at<uchar>(i, j);
					min_num = std::min(min_num, val_roi);
				}
			}
			dark_channel_mat.at<uchar>(m - border, n - border) = min_num;
		}
	}
	return dark_channel_mat;
}


int calculate_A(cv::Mat src, cv::Mat dark_channel_mat)
{
	std::vector<cv::Mat> rgbChannels(3);
	split(src, rgbChannels);
	map<int, cv::Point> pair_data;
	map<int, cv::Point>::iterator iter;
	std::vector<cv::Point> cord;
	int max_val = 0;
	//cout << dark_channel_mat.rows << " " << dark_channel_mat.cols << endl;
	for (int i = 0; i < dark_channel_mat.rows; i++)
	{
		for (int j = 0; j < dark_channel_mat.cols; j++)
		{
			int val = dark_channel_mat.at<uchar>(i, j);
			cv::Point pt;
			pt.x = j;
			pt.y = i;
			pair_data.insert(make_pair(val, pt));
		}
	}

	for (iter = pair_data.begin(); iter != pair_data.end(); iter++)
	{
		//cout << iter->first << endl;
		cord.push_back(iter->second);
	}
	for (int m = 0; m < cord.size(); m++)
	{
		cv::Point tmp = cord[m];
		int val_1, val_2, val_3;
		val_1 = rgbChannels[0].at<uchar>(tmp.y, tmp.x);
		val_2 = rgbChannels[1].at<uchar>(tmp.y, tmp.x);
		val_3 = rgbChannels[2].at<uchar>(tmp.y, tmp.x);
		max_val = std::max(val_1, val_2);
		max_val = std::max(max_val, val_3);
	}
	return max_val;
}


cv::Mat calculate_tx(cv::Mat& src, int A, cv::Mat& dark_channel_mat)
{
	cv::Mat dst;//����������t(x)
	cv::Mat tx;
	float dark_channel_num;
	dark_channel_num = A / 255.0;
	dark_channel_mat.convertTo(dst, CV_32FC3, 1 / 255.0);//��������t(x)
	dst = dst / dark_channel_num;
	tx = 1 - 0.95 * dst;//���յ�txͼ

	return tx;
}


cv::Mat haze_removal_img(cv::Mat& src, int A, cv::Mat& tx)
{
	cv::Mat result_img(src.rows, src.cols, CV_8UC3);
	vector<cv::Mat> srcChannels(3), resChannels(3);
	split(src, srcChannels);
	split(result_img, resChannels);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			for (int m = 0; m < 3; m++)
			{
				int value_num = srcChannels[m].at<uchar>(i, j);
				float max_t = tx.at<float>(i, j);
				if (max_t < 0.1)
				{
					max_t = 0.1;
				}
				resChannels[m].at<uchar>(i, j) = (value_num - A) / max_t + A;
			}
		}
	}
	merge(resChannels, result_img);

	return result_img;
}
//*********��������������������������������������ȥ���㷨����Ҫ���� ����������������������**************//





