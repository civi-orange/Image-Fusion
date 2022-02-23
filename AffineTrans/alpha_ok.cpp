#include "funcHead.h"

using namespace cv;
using namespace std;
//5
int main5(int argc, char** argv)
{
	// 输入
	Mat imsrc1 = imread("./image/IR_18rad.bmp");
	Mat imsrc2 = imread("./image/VIS_18dhvR.bmp");
	imshow("1", imsrc1);
	imshow("2", imsrc2);
	clock_t time_start = clock();
	resize(imsrc2, imsrc2, imsrc1.size());
	//cout << imsrc1.size() << endl;
	Mat imsrc2_scaler;      //将imsrc2缩放到imsrc1的尺寸
	Mat imdst;              //合成的目标图像

	double alpha = 0.8;
	double gamma = 0;

	if (imsrc1.size() != imsrc2.size()) {
		printf("resize start!\n");
		//resize(imsrc2, imsrc2_scaler, Size(imsrc1.cols, imsrc1.rows), 0, 0, INTER_LINEAR);
		resize(imsrc2, imsrc2_scaler, imsrc1.size(), 0, 0, INTER_LINEAR);    //方法2
		addWeighted(imsrc1, alpha, imsrc2_scaler, 1 - alpha, gamma, imdst);
	}
	else {
		addWeighted(imsrc1, alpha, imsrc2, 1 - alpha, gamma, imdst);
	}
	clock_t time_end = clock();
	cout << "time:" << time_end - time_start << "ms" << endl;
	imshow("imdst", imdst);
	cv::waitKey();
	return 0;
}