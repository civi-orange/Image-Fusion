#include "funcHead.h"

//windows 读取指定文件夹下文件名字  出现此警告 请在预编译文件起始增加#define _CRT_SECURE_NO_DEPRECATE
void listFiles(string dir, vector<string>& files, string str_img_type)
{
	string filename;
	filename.assign(dir).append("*."+ str_img_type);//匹配文件类型
	intptr_t handle;
	_finddata_t findData;//读取文件名
	handle = _findfirst(filename.c_str(), &findData);
	if (handle == -1)        // 检查是否成功
		return;
	do
	{
		string xxx = dir;
		xxx.append(findData.name);
		files.push_back(xxx);
		//files.push_back(filename);
	} while (_findnext(handle, &findData) == 0);
	_findclose(handle);    // 关闭搜索句柄
}


void CalcCorners(const cv::Mat& H, const cv::Mat& src, four_corners_t& corners)//将H*SRC的新图角点坐标
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	//cout << "V2: " << endl << V2 << endl;
	//cout << "V1: " << endl << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

	return;
}


void RegionGrow(cv::Mat src, cv::Mat& matDst)
{
	int DIR[8][2] = { { -1, -1 },{ 0, -1 },{ 1, -1 },{ 1, 0 },{ 1, 1 },{ 0, 1 },{ -1, 1 },{ -1, 0 } };//生长方向顺序数据//loop
	cv::Point2i pt = cv::Point(0, src.rows);//起始位置根据，生长方向找出 根据loop调整使代码时间复杂付最低
	std::vector<cv::Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push_back(pt);							//将生长点压入栈中
	
	double min_g, max_g;
	minMaxLoc(src, &min_g, &max_g);
	int iThre = (max_g + 4 * min_g) / 5;						//定义区域灰度差阈值
	

	int nSrcValue = 0;								//生长起点灰度值，起始灰度定为中值灰度
	int nCurValue = 0;								//当前生长点灰度值

	matDst = cv::Mat(src.size(), CV_8UC1, cv::Scalar(255));	//创建一个空白区域，填充为黑色												
	matDst.at<uchar>(pt.y, pt.x) = 0;				//标记生长点

	//nSrcValue = src.at<uchar>(pt.y, pt.x);		  //记录生长点的灰度值，不以图像为基准，以中值为基准

	nSrcValue = mean(src)[0];						//其实灰度定位灰度可行域的1/3处
	
	cv::Point2i ptGrowing;							 //待生长点位置	
	while (!vcGrowPt.empty())						//生长栈不为空则生长
	{
		pt = vcGrowPt.back();						//取出一个生长点
		vcGrowPt.pop_back();
		//分别对八个方向上的点进行生长
		for (int i = 0; i < 8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点//loop
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;
			int nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值
			if (nGrowLable == 255)					//如果标记点还没有被生长
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				//if (abs(nSrcValue - nCurValue) < th)					//在阈值范围内则生长
				if (nCurValue - nSrcValue < iThre)	//修改为大于阈值？
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 0;		//标记为白色
					vcGrowPt.push_back(ptGrowing);					//将下一个生长点压入栈中
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
		pyrUp(img_up, up, cv::Size(img_up.cols * 2, img_up.rows * 2));//顶层上采样 
		img_lp = img_sub - up;
		Img_Laplacian_pyr.push_back(img_lp);
	}

	return Img_Laplacian_pyr;
}

std::vector<cv::Mat> Fusion_Laplace_Pyr(std::vector<cv::Mat>& Img_front_lp, std::vector<cv::Mat>& Img_back_lp, std::vector<cv::Mat>& mask_gau, std::vector<cv::Mat>& blend_lp)
{

	int level = Img_front_lp.size();

	for (int i = 0; i < level; i++)//0为大图即金字塔底层                                        
	{
		cv::Mat A = Img_front_lp[i].mul(mask_gau[i]); //Alpha（Attention:）Data types must be unified 
		cv::Mat antiMask;
		if (mask_gau[0].channels() == 3) {
			antiMask = cv::Scalar(1.0, 1.0, 1.0) - mask_gau[i];//彩色图
		}
		else {
			antiMask = cv::Scalar(1.0) - mask_gau[i];//灰度图
		}
		cv::Mat B = Img_back_lp[i].mul(antiMask); //Beta
		cv::Mat blendedLevel = A + B;
		blend_lp.push_back(blendedLevel);
	}
	return blend_lp;
}

int Laplace_pyramid_fusion(cv::Mat& frame_back, cv::Mat& frame_front, double dAlpha, int pyr_level, cv::Mat& output)
{
	//融合策略0
	cv::Mat mask = cv::Mat::zeros(frame_front.size(), CV_32FC1);
	mask(cv::Range::all(), cv::Range::all()) = dAlpha;//多尺度线性融合
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

	cv::Mat img_start = Gau_Pyr_front.back().mul(mask_Pyr.back()) + ((Gau_Pyr_back.back()).mul(cv::Scalar(1.0) - mask_Pyr.back()));//灰度图

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
	//提取特征点    
	cv::Ptr<cv::FastFeatureDetector> Detector = cv::FastFeatureDetector::create(50);  //阈值 
	vector<cv::KeyPoint> keyPoint_front, keyPoint_back;
	Detector->detect(img_front, keyPoint_front);
	Detector->detect(img_back, keyPoint_back);

	//特征点描述，为下边的特征点匹配做准备    
	cv::Ptr<cv::SiftDescriptorExtractor> SiftDescriptor = cv::SiftDescriptorExtractor::create();
	cv::Mat imageDesc_front, imageDesc_back;
	SiftDescriptor->compute(img_front, keyPoint_front, imageDesc_front);
	SiftDescriptor->compute(img_back, keyPoint_back, imageDesc_back);

	//BruteForceMatcher 
	//cv::Ptr<cv::BFMatcher> bfmatcher = cv::BFMatcher::create(cv::NORM_L2, true);
	cv::BFMatcher bfmatcher;
	vector<vector<cv::DMatch> > matchePoints;
	vector<cv::DMatch> GoodMatchePoints;

	vector<cv::Mat> train_desc_fast(1, img_front);//定义1个元素都为imageDesc1
	bfmatcher.add(train_desc_fast);
	bfmatcher.train();
	bfmatcher.knnMatch(imageDesc_back, matchePoints, 2);
	//cout << "total match points: " << matchePoints.size() << endl;

	// Lowe's algorithm,获取优秀匹配点
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

	//图像匹配
	vector<cv::Point2f> imagePoints_front, imagePoints_back;
	for (int i = 0; i < GoodMatchePoints.size(); i++)
	{
		imagePoints_back.push_back(keyPoint_back[GoodMatchePoints[i].queryIdx].pt);//queryidx为图2
		imagePoints_front.push_back(keyPoint_front[GoodMatchePoints[i].trainIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	homo = findHomography(imagePoints_front, imagePoints_back, cv::RANSAC);  // src = h * dst
	////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵

	four_corners_t corners;
	//计算配准图的四个顶点坐标
	CalcCorners(homo, img_front, corners);//将frame_front的角点坐标通过homo映射到frame_back的四个角点

	//图像配准  
	cv::Mat imageTransform1, imageTransform2;
	warpPerspective(img_front, imageTransform1, homo, cv::Size(MAX(corners.right_top.x, corners.right_bottom.x), img_back.rows));//将frame_1向frame_2上做映射，以frame_2的rows为基准
	//warpPerspective(frame_1, imageTransform1, homo, cv::Size(MAX(corners.right_top.x, corners.right_bottom.x), MAX(corners.right_top.y, corners.right_bottom.y)));
	//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	imshow("直接经过透视矩阵变换", imageTransform1);


	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
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

//优化两图的连接处，使得拼接自然 //重叠区域做ALPHA融合
void OptimizeSeam(cv::Mat& img1, cv::Mat& trans, four_corners_t& corners, cv::Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  
	double processWidth = MIN(img1.cols, trans.cols) - start;//重叠区域的宽度  ，取img1的原因？？？如果tran.cols>img.cols无可厚非；如果tran.cols<img.cols
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重
	if (img1.channels() == 3)
	{
		for (int i = 0; i < rows; i++)
		{
			uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
			uchar* t = trans.ptr<uchar>(i);
			uchar* d = dst.ptr<uchar>(i);
			for (int j = start; j < cols; j++)
			{
				//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
				if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
				{
					alpha = 1;
				}
				else
				{
					//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
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
			uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
			uchar* t = trans.ptr<uchar>(i);
			uchar* d = dst.ptr<uchar>(i);
			for (int j = start; j < cols; j++)
			{
				//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
				if (t[j] == 0)
				{
					alpha = 1;
				}
				else
				{
					//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
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
	
	//统计灰度级个数
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

//2022/01/18 添加工程本文件没有这个函数
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
	//统计灰度级个数
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
			Dist[Y] += abs(Y - X) * HistGram[X];//论文公式（9），灰度的距离只有绝对值，这里其实可以优化速度，但计算量不大，没必要了  
		max_gray = max(max_gray, Dist[Y]);
		min_gray = min(min_gray, Dist[Y]);
	}

	int max_min = max_gray - min_gray;

	for (Y = 0; Y < row; Y++)
	{
		for (X = 0; X < col; X++)
		{
			dst.at<uchar>(Y, X) = (Dist[gray.at<uchar>(Y, X)] - min_gray) * 255 / max_min;//计算全图每个像素的显著性 
			//Sal.at<uchar>(Y,X) = (Dist[gray.at<uchar>(Y, X)])*255/(max_gray);//计算全图每个像素的显著性  
		}
	}

}

//灰度拉伸
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
		resize(src2, src2_1, src1.size(), 0, 0, cv::INTER_LINEAR);    //方法2
		addWeighted(src1, dAlpha, src2_1, 1 - dAlpha, gamma, dst);
	}
	else {
		addWeighted(src1, dAlpha, src2, 1 - dAlpha, gamma, dst);
	}
	return true;
}

//仿射变换
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
		dst_rows = round(fabs(param.y_scale * src.rows * cos(Angle)) + fabs(param.x_scale * src.cols * sin(Angle)));//再经过旋转后图像高度
		dst_cols = round(fabs(param.x_scale * src.cols * cos(Angle)) + fabs(param.y_scale * src.rows * sin(Angle)));//再经过旋转后图像宽度
		mat_trans_temp.at<double>(0, 2) += ((dst_cols - src.cols) / 2 - param.x_offset);//平移显示全图
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
				// 获取像素
				int b = src.at<cv::Vec3b>(row, col)[0];
				int g = src.at<cv::Vec3b>(row, col)[1];
				int r = src.at<cv::Vec3b>(row, col)[2];
				// 调整亮度和对比度
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
				// 获取像素
				int scalar = src.at<uchar>(row, col);

				// 调整亮度和对比度
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
				dst.at<uchar>(row, col) = (uchar)table[src.at<uchar>(row, col)];//查表法，减少计算量	
			}
		}
	}

	return true;
}

//图像熵
float entropy_a(cv::Mat& src)
{
	//图像一维熵
	cv::Mat hist;
	const int histSize[] = { 256 }; //直方图每一个维度划分的柱条的数目
	float pranges[] = { 0,256 };
	const float* ranges[] = { pranges };
	calcHist(&src, 1, 0, cv::Mat(), hist, 1, histSize, ranges, true, false);//计算直方图	
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

//图像联合熵
float entropy_ab(cv::Mat& src1, cv::Mat& src2)
{
	//图像联合熵
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

//图像二维熵
float entropy_2(cv::Mat& src)
{
	//图像二维熵
	cv::Mat kernel = (cv::Mat_<double>(3, 3) << 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 8, 0.0, 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 8);
	cv::Mat gray_filter;
	filter2D(src, gray_filter, -1, kernel);
	double _entropy = 0.0;
	_entropy = entropy_ab(src, gray_filter);
	return _entropy;
}

//图像互信息MI
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
		//return (EN_1f + EN_2f) / (EN_1 + EN_2);//融合互信息；而两幅图的互信息=EN1+EN2-EN12
		return EN_1 + EN_2 + 2 * EN_f - EN_1f - EN_2f;
	}
	else
	{
		return EN_1 + EN_2 - EN_12;
	}
}

//图像平均梯度
float mean_grad(cv::Mat& src)
{
	//图像平均梯度
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

//SSIM结构化相似度
float calc_SSIM(cv::Mat& src, cv::Mat& fuse)
{
	//可分区域进行计算，此处直接计算

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

//空间频率
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

//相关系数
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

//边缘信息保存量
float calc_QAB_F(cv::Mat& src1, cv::Mat& src2, cv::Mat& fuse)
{

	return -1;
}

//画图像直方图
void hist_graph(cv::Mat hist)
{
	int scale = 2;
	int hist_height = 256;
	cv::Mat hist_img = cv::Mat::zeros(hist_height, 256 * scale, CV_8UC3); //创建一个黑底的8位的3通道图像，高256，宽256*2
	double max_val;
	minMaxLoc(hist, 0, &max_val, 0, 0);//计算直方图的最大像素值
	//将像素的个数整合到 图像的最大范围内
	//遍历直方图得到的数据
	for (int i = 0; i < 256; i++)
	{
		float bin_val = hist.at<float>(i);   //遍历hist元素（注意hist中是float类型）
		int intensity = cvRound(bin_val * hist_height / max_val);  //绘制高度
		rectangle(hist_img, cv::Point(i * scale, hist_height - 1), cv::Point((i + 1) * scale - 1, hist_height - intensity), cv::Scalar(255, 255, 255));//绘制直方图
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
	output.fusion_Qab_f = -1;//未实现
	//VIF
	output.fusion_VIF = -1;//未实现

	return true;
}


cv::Mat GuidedFilter(cv::Mat& input, cv::Mat& guidImg, int r, double eps)
{	
	int wsize = 2 * r + 1;	
	cv::Mat I, p;
	//数据类型转换
	guidImg.convertTo(I, CV_64F, 1.0 / 255.0);
	input.convertTo(p, CV_64F, 1.0 / 255.0);


	//meanI=fmean(I)
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	//meanP=fmean(P)
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//corrI=fmean(I.*I)
	cv::Mat mean_II;
	mean_II = I.mul(I);
	cv::boxFilter(mean_II, mean_II, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	//corrIp=fmean(I.*p)
	cv::Mat mean_Ip;
	mean_Ip = I.mul(p);
	cv::boxFilter(mean_Ip, mean_Ip, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波


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
	cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//q=meana.*I+meanb
	cv::Mat out;
	out = mean_a.mul(I) + mean_b;

	//数据类型转换
	//I.convertTo(guidImg, CV_8U, 255);
	//p.convertTo(input, CV_8U, 255);
	out.convertTo(out, CV_8U, 255);

	return out;

}

//图像 对比显示
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








//最大熵分割算法 二值分割，p是调节因子手动
int Max_Entropy(cv::Mat& src, cv::Mat& dst, int thresh, int p) 
{
	const int Grayscale = 256;
	int Graynum[Grayscale] = { 0 };
	int r = src.rows;
	int c = src.cols;
	for (int i = 0; i < r; ++i) {
		const uchar* ptr = src.ptr<uchar>(i);
		for (int j = 0; j < c; ++j) {
			if (ptr[j] == 0)				//排除掉黑色的像素点
				continue;
			Graynum[ptr[j]]++;
		}
	}

	float probability = 0.0; //概率
	float max_Entropy = 0.0; //最大熵
	int totalpix = r * c;
	for (int i = 0; i < Grayscale; ++i) {

		float HO = 0.0; //前景熵
		float HB = 0.0; //背景熵

		//计算前景像素数
		int frontpix = 0;
		for (int j = 0; j < i; ++j) {
			frontpix += Graynum[j];
		}
		//计算前景熵
		for (int j = 0; j < i; ++j) {
			if (Graynum[j] != 0) {
				probability = (float)Graynum[j] / frontpix;
				HO = HO + probability * log(1 / probability);
			}
		}

		//计算背景熵
		for (int k = i; k < Grayscale; ++k) {
			if (Graynum[k] != 0) {
				probability = (float)Graynum[k] / (totalpix - frontpix);
				HB = HB + probability * log(1 / probability);
			}
		}

		//计算最大熵
		if (HO + HB > max_Entropy) {
			max_Entropy = HO + HB;
			thresh = i + p;
		}
	}

	//阈值处理
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


///去雾算法的 实现示例
//int main()
//{
//	cv::Mat src = cv::imread("C:/Users/18301/Desktop/1.png");
//	cv::Mat dst;
//	cvtColor(src, dst, cv::COLOR_BGR2GRAY);
//	cv::Mat dark_channel_mat = dark_channel(src);//输出的是暗通道图像
//	int A = calculate_A(src, dark_channel_mat);
//	cv::Mat tx = calculate_tx(src, A, dark_channel_mat);
//	cv::Mat tx_ = guidedfilter(dst, tx, 30, 0.001);//导向滤波后的tx，dst为引导图像
//	cv::Mat haze_removal_image;
//	haze_removal_image = haze_removal_img(src, A, tx_);
//	cv::namedWindow("去雾后的图像", 0);
//	cv::namedWindow("原始图像", 0);
//	imshow("原始图像", src);
//	imshow("去雾后的图像", haze_removal_image);
//	cv::waitKey(0);
//	return 0;
//}

//*********↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓去雾算法的主要代码 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*********//
//导向滤波，用来优化t(x)，针对单通道
cv::Mat guidedfilter(cv::Mat& srcImage, cv::Mat& srcClone, int r, double eps)
{
	//转换源图像信息
	srcImage.convertTo(srcImage, CV_32FC1, 1 / 255.0);
	srcClone.convertTo(srcClone, CV_32FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	cv::Mat boxResult;
	//步骤一：计算均值
	boxFilter(cv::Mat::ones(nRows, nCols, srcImage.type()), boxResult, CV_32FC1, cv::Size(r, r));
	//生成导向均值mean_I
	cv::Mat mean_I;
	boxFilter(srcImage, mean_I, CV_32FC1, cv::Size(r, r));
	//生成原始均值mean_p
	cv::Mat mean_p;
	boxFilter(srcClone, mean_p, CV_32FC1, cv::Size(r, r));
	//生成互相关均值mean_Ip
	cv::Mat mean_Ip;
	boxFilter(srcImage.mul(srcClone), mean_Ip, CV_32FC1, cv::Size(r, r));
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	//生成自相关均值mean_II
	cv::Mat mean_II;
	//应用盒滤波器计算相关的值
	boxFilter(srcImage.mul(srcImage), mean_II, CV_32FC1, cv::Size(r, r));
	//步骤二：计算相关系数
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);
	cv::Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
	//步骤三：计算参数系数a,b
	cv::Mat a = cov_Ip / (var_I + eps);
	cv::Mat b = mean_p - a.mul(mean_I);
	//步骤四: 计算系数 a/b 的均值
	cv::Mat mean_a;
	boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
	mean_a = mean_a / boxResult;
	cv::Mat mean_b;
	boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));
	mean_b = mean_b / boxResult;
	//步骤五：生成输出矩阵
	cv::Mat resultMat = mean_a.mul(srcImage) + mean_b;
	return resultMat;
}

//计算暗通道图像矩阵，针对三通道彩色图像
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
	cv::Mat dst;//是用来计算t(x)
	cv::Mat tx;
	float dark_channel_num;
	dark_channel_num = A / 255.0;
	dark_channel_mat.convertTo(dst, CV_32FC3, 1 / 255.0);//用来计算t(x)
	dst = dst / dark_channel_num;
	tx = 1 - 0.95 * dst;//最终的tx图

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
//*********↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑去雾算法的主要代码 ↑↑↑↑↑↑↑↑↑↑↑**************//





