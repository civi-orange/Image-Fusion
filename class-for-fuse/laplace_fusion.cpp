#include "Fusion_Image.h"

#define IMAGES_PATH			""
//#define IMAGES_PATH			"../../AffineTrans/image/Tree_sequence/"
//#define IMAGES_PATH			"../../AffineTrans/image/Duine_sequence/"
//#define IMAGES_PATH			"../../AffineTrans/image/Nato_camp_sequence/"
//#define IMAGES_PATH		"../../AffineTrans/image/city/"


int main()
{	





	// 初始化类方法，融合比例设置
	FIMG::method_group method_f;
	FIMG::Fusion_Image ffffffff;
	FIMG::Fusion_Image_Param pppp;
	pppp.dAlpha = 0.6;
	ffffffff.init_parameter(pppp);

	cv::Mat frame = cv::imread("D:/WorkData/dataset/ltir_v1_0_8bit/crossing/00000007.png");
	cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
	cv::Mat frame_;
	if (mean(frame)[0] > 128)
	{
		method_f.Adjust_gamma(frame, frame_, 2);
	}
	else
	{
		frame.copyTo(frame_);
	}
	cv::Mat frame1;
	int xxx = method_f.Max_Entropy(frame_, frame1,-5);
	imshow("what", frame1);

	//主程序
	vector<cv::String> vct_vis_files;
	vector<cv::String> vct_ir_files;								 
	method_f.listFiles(string(IMAGES_PATH).append("visual/"), vct_vis_files, "bmp");
	method_f.listFiles(string(IMAGES_PATH).append("thermal/"), vct_ir_files, "bmp");

	if (vct_vis_files.size() == 0)
	{
		cv::String ir = "D:/WorkData/dataset/TNO_Image_Fusion_Dataset/Athena_images/soldier_behind_smoke_2/IR_meting012-1500_g.bmp"; //"D:/WorkData/dataset/TNO_Image_Fusion_Dataset/DHV_images/bench/IR_37rad.bmp";
		vct_ir_files.push_back(ir);
		cv::String vis = "D:/WorkData/dataset/TNO_Image_Fusion_Dataset/Athena_images/soldier_behind_smoke_2/VIS_meting012-1500_r.bmp"; //"D:/WorkData/dataset/TNO_Image_Fusion_Dataset/DHV_images/bench/VIS_37dhvR.bmp";
		vct_vis_files.push_back(vis);

	}

	for (int i = 0; i < vct_vis_files.size(); i++)
	{

		//获取图像
		cv::Mat frame_front, frame_back;
		//cv::VideoCapture cap;
		//cap.open("");
		frame_front = cv::imread(vct_ir_files[i]);
		frame_back = cv::imread(vct_vis_files[i]);
		method_f.GrayStretch(frame_back, frame_back);
		method_f.GrayStretch(frame_front, frame_front);
		resize(frame_front, frame_front, cv::Size(640, 512)); //以此为基准
		resize(frame_back, frame_back, frame_front.size());

		//灰度化
		if (frame_front.channels() == 3)
		{
			cvtColor(frame_front, frame_front, cv::COLOR_BGR2GRAY);
		}
		if (frame_back.channels() == 3)
		{
			cvtColor(frame_back, frame_back, cv::COLOR_BGR2GRAY);
		}

		//图像gamma校正，目前为解决图像过曝问题,“曝光”不足时待实验
		cv::Mat front_gamma, back_gamma;
		if (mean(frame_front)[0] > 128)
		{
			method_f.Adjust_gamma(frame_front, front_gamma, 2);
		}
		else
		{
			frame_front.copyTo(front_gamma);
		}
		if (0/*mean(frame_back)[0] > 128*/)
		{
			method_f.Adjust_gamma(frame_back, back_gamma, 1);
		}
		else
		{
			frame_back.copyTo(back_gamma);
		}

		if (pppp.init_flag == false)
		{
			ffffffff.init_fusion(front_gamma);
			pppp.init_flag = true;
		}
		//imshow("f_g", front_gamma);

		cv::Mat target_target, mat_front, frame_out;
		mat_front = cv::Mat(front_gamma.size(), CV_8UC1, cv::Scalar(50));

		//target ROI 
		clock_t time_k1 = clock();	cv::Mat guit_mat;
		cv::ximgproc::guidedFilter(front_gamma, front_gamma, guit_mat, 3, 500);
		clock_t time_kg = clock();
		

		int thresh = method_f.Max_Entropy(guit_mat, target_target, 3, 3);
		clock_t time_kme = clock();

		//红外微光图像ROI融合
		cv::Mat out_or = cv::Mat::zeros(front_gamma.size(),front_gamma.type());
		for (int x = 0; x < back_gamma.rows; ++x) 
		{
			for (int y = 0; y < back_gamma.cols; ++y) 
			{
				int fgray = frame_front.at<uchar>(x, y);
				int bgray = frame_back.at<uchar>(x, y);
				int oooo = target_target.at<uchar>(x, y);
				if (oooo == 255)
				{
					if (fgray < bgray)
					{
						out_or.at<uchar>(x, y) = (uchar)bgray;
					}
					else
					{
						out_or.at<uchar>(x, y) = (uchar)fgray;
					}
				}
				else
				{
					out_or.at<uchar>(x, y) = (uchar)bgray;
				}
			}
		}
		clock_t time_ooooo = clock();


		//ROI pyramid 权值融合
		back_gamma.copyTo(frame_out);
		frame_front.copyTo(mat_front, target_target);
		cv::Mat fusion_image_k;
		ffffffff.fusion_image(mat_front, frame_out, fusion_image_k);
		fusion_image_k.copyTo(frame_out, target_target);
		clock_t time_k3 = clock();

		std::cout << "openCV guide-Filter: " << time_kg - time_k1 << "ms" << std::endl;
		std::cout << "image segment: " << time_kme - time_kg << "ms" << std::endl;
		std::cout << "region fuse: " << time_ooooo - time_k1 << "ms" << std::endl;
		std::cout << "roi fusion lp: " << time_k3 - time_k1 << "ms" << std::endl;


		//imshow("guit_mat", guit_mat);
		putText(out_or, std::to_string(time_ooooo - time_k1) + " ms", cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0), 1);

		//全图 pyramid 权值融合------------------------------------------------------------------
		clock_t time_1 = clock();
		int pyr_level = 6;
		int temp_int = std::pow(2, pyr_level);
		//the redundant code
		while (frame_front.cols % temp_int != 0 || frame_front.rows % temp_int != 0)
		{
			//std::cout<<"The pyramid has too many layers!"<< std::endl;
			temp_int = pow(2, pyr_level--);
		}
		cv::Mat fusion_image;	
		ffffffff.fusion_image(front_gamma, back_gamma, fusion_image);
		clock_t time_end = clock();		
		std::cout << "the full fusion lp: " << time_end - time_1 << "ms" << std::endl;
		std::cout << "-------------------------------------------" << std::endl;
		//imshow("temp", front_gamma);
		putText(frame_out, std::to_string(time_k3 - time_k1) + " ms", cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0), 1);
		putText(fusion_image, std::to_string(time_end - time_1) + " ms", cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0), 1);
		
		
		//引导滤波显著融合，由于数据集微光图像纹理不显著，红外边缘信息显著，结果基本只有红外信息，暂不使用
		//cv::Mat out_gfffff, out_gfffff1;
		//ffffffff.GFF_Fusion(frame_front, frame_back, out_gfffff);
		//method_f.GrayStretch(out_gfffff, out_gfffff);
		//imshow("out_gfffff", out_gfffff);


		//综合显示图像----------------------------------------------------------------
		vector<cv::Mat> vct_show_images;
		vct_show_images.push_back(frame_back);
		vct_show_images.push_back(frame_front);
		vct_show_images.push_back(target_target);
		vct_show_images.push_back(fusion_image);
		vct_show_images.push_back(frame_out);
		vct_show_images.push_back(out_or);
		cv::Mat allimage = method_f.All_Images_inWindow(vct_show_images, 3, 2);
		imshow("IR--VIS--ROI--Weight", allimage);


		//对比融合图像保存------------------------------------------------------
		string str_img = string(IMAGES_PATH).append("fuse/");
		if (_access(str_img.c_str(), 0) == -1)	//如果文件夹不存在则创建
			auto xx = _mkdir(str_img.c_str());	
		//imwrite(str_img + to_string(i) + ".bmp", allimage);

		//融合图图像指标
		FIMG::fusion_image_index k_fuse, w_fuse;
		ffffffff.Get_Fusion_indicator(frame_back, frame_front, out_or, k_fuse);
		ffffffff.Get_Fusion_indicator(frame_back, frame_front, fusion_image, w_fuse);
		
		 ////融合指标对比数据 保存
		 ////string file_path = getCurFilePath();//获取当前文件绝对路径
		 //string pathname = "./FileData/";//相对路径
		 //time_t t = time(0);
		 //char ch[64];
		 ////strftime(ch, sizeof(ch), "%Y-%m-%d-%H-%M-%S", localtime(&t)); //年-月-日 时-分-秒
		 //strftime(ch, sizeof(ch), "%Y-%m-%d-%H-%M", localtime(&t)); //年-月-日 时-分
		 //std::ofstream test_value(pathname + "k_fuse_and_w_fuse" + ch + ".txt", std::ios::app | std::ios::out);
		 //test_value <<" k_fuse  " << "    w_fuse  " << std::endl;			
		 //test_value <<k_fuse.fusion_EN << "	" << w_fuse.fusion_EN << std::endl;
		 //test_value <<k_fuse.fusion_MI << "	" << w_fuse.fusion_MI << std::endl;
		 //test_value <<k_fuse.fusion_meanGrad << "	" << w_fuse.fusion_meanGrad << std::endl;
		 //test_value <<k_fuse.fusion_SSIM_irf << "	" << w_fuse.fusion_SSIM_irf << std::endl;
		 //test_value <<k_fuse.fusion_SSIM_visf << "	" << w_fuse.fusion_SSIM_visf << std::endl;
		 //test_value <<k_fuse.fusion_SF << "	" << w_fuse.fusion_SF << std::endl;
		 //test_value <<k_fuse.fusion_SD << "	" << w_fuse.fusion_SD << std::endl;
		 //test_value <<k_fuse.fusion_CC_irf << "	" << w_fuse.fusion_CC_irf << std::endl;
		 //test_value <<k_fuse.fusion_CC_visf << "	" << w_fuse.fusion_CC_visf << std::endl;
		 ////test_value <<k_fuse.fusion_VIF << "	" << w_fuse.fusion_VIF << std::endl;
		 ////test_value <<k_fuse.fusion_Qab_f << "	" << w_fuse.fusion_Qab_f		<< std::endl;
		 //test_value << "---------------------------------------------" << std::endl;
		 //test_value.close();

		cv::waitKey();
	}

	cv::destroyAllWindows();
	return true;
}





