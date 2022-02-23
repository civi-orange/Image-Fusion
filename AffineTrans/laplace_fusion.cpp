#include "funcHead.h"

#include "Fusion_Image.h"

//#define IMAGES_PATH			""
#define IMAGES_PATH		"./image/Tree_sequence/"
//#define IMAGES_PATH		"./image/Duine_sequence/"
//#define IMAGES_PATH		"./image/Nato_camp_sequence/"
//#define IMAGES_PATH		"./image/city/"



//白天,过曝的情况，效果不好
int main()
{
	////测试分割算法
	//cv::Mat image = cv::imread("D:/WorkData/dataset/TNO_Image_Fusion_Dataset/DHV_images/bench/IR_37rad.bmp");
	//cvtColor(image, image, cv::COLOR_BGR2GRAY);
	//cv::Mat mat_ir;
	//Max_Entropy(image, mat_ir, 0, 10);
	//imshow("ir_enhance", mat_ir);
	//cv::waitKey();

	//主程序
	vector<cv::String> vct_vis_files;
	vector<cv::String> vct_ir_files;								 
	listFiles(string(IMAGES_PATH).append("visual/"), vct_vis_files, "bmp");
	listFiles(string(IMAGES_PATH).append("thermal/"), vct_ir_files, "bmp");

	if (vct_vis_files.size() == 0)
	{
		cv::String ir = "D:/WorkData/dataset/TNO_Image_Fusion_Dataset/DHV_images/bench/IR_37rad.bmp";
		vct_ir_files.push_back(ir);
		cv::String vis = "D:/WorkData/dataset/TNO_Image_Fusion_Dataset/DHV_images/bench/VIS_37dhvR.bmp";
		vct_vis_files.push_back(vis);


		//cv::Mat frame = cv::imread("./image/two/soldier1/meting012-1200_rg.bmp");
		//cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		//imshow("what", frame);
	}

	// 初始化类方法，
	FIMG::Fusion_Image ffffffff;
	FIMG::Fusion_Image_Param pppp;
	ffffffff.init_parameter(pppp);


	for (int i = 0; i < vct_vis_files.size(); i++)
	{

		//获取图像
		cv::Mat frame_front, frame_back;
		//cv::VideoCapture cap;
		//cap.open("");
		frame_front = cv::imread(vct_ir_files[i]);
		frame_back = cv::imread(vct_vis_files[i]);
		GrayStretch(frame_back, frame_back);
		GrayStretch(frame_front, frame_front);		
		resize(frame_front, frame_front, cv::Size(640, 512)); //以此为基准
		resize(frame_back, frame_back, frame_front.size());



		//通道转化
		if (frame_front.channels() == 3)
		{
			cvtColor(frame_front, frame_front, cv::COLOR_BGR2GRAY);
		}
		if (frame_back.channels() == 3)
		{
			cvtColor(frame_back, frame_back, cv::COLOR_BGR2GRAY);
		}




		// 测试类方法有效性
		{	
			clock_t time_oo1 = clock();
			if (pppp.init_flag == false)
			{
				ffffffff.init_fusion(frame_front);
				pppp.init_flag = true;
			}
			cv::Mat oooooo;
			ffffffff.fusion_image(frame_front, frame_back, oooooo);
			clock_t time_oo2 = clock();
			std::cout << "oo time:" << time_oo2 - time_oo1 << "ms" << std::endl;
			//imshow("oooo", oooooo);

		}

			clock_t time_k1 = clock();
			cv::Mat Target_mat, mat_front, frame_out;
			mat_front = cv::Mat(frame_front.size(), CV_8UC1, cv::Scalar(50));
			//抠图融合
			cv::Mat guit_mat = GuidedFilter(frame_front, frame_front, 10, 0.001);
			Get_Infrared_target_region(guit_mat, Target_mat);
			//imshow("Target_mat", Target_mat);
			frame_back.copyTo(frame_out);
			frame_front.copyTo(mat_front, Target_mat);

			cv::Mat frame_out11;
			frame_back.copyTo(frame_out11);
			mat_front.copyTo(frame_out11, Target_mat);
			imshow("frame_out_111", frame_out11);

			//imshow("mat_front", mat_front);
			//GaussianBlur(mat_front, mat_front, cv::Size(3, 3), 3);
			//imshow("mat_front1", mat_front);
			//blur(mat_front, mat_front, cv::Size(9, 9));
			/*
			cv::Mat fusion_image_k;
			Laplace_pyramid_fusion(mat_front, frame_out, 0.8, 6, fusion_image_k);*/
			cv::Mat fusion_image_k;
			ffffffff.fusion_image(mat_front, frame_out, fusion_image_k);
			//Alpha_Beta_Image_fuse(mat_back, mat_front, 0.1, 0, fusion_image_k);//融合图像增加背景
			//add(frame_back, fusion_image_k, fusion_image_k);
			fusion_image_k.copyTo(frame_out, Target_mat);
			clock_t time_k2 = clock();
			std::cout << "k time:" << time_k2 - time_k1 << "ms" << std::endl;
			//imshow("fusion_image_k", fusion_image_k);	
			
			


		//权值融合
		clock_t time_1 = clock();
		int pyr_level = 6;
		int temp_int = std::pow(2, pyr_level);
		//the redundant code
		while (frame_front.cols % temp_int != 0 || frame_front.rows % temp_int != 0)
		{
			//std::cout<<"The pyramid has too many layers!"<< std::endl;
			temp_int = pow(2, pyr_level--);
		}
		
		//融合图像
		cv::Mat fusion_image, mat_temp;
		clock_t time_start = clock();

		//mat_temp = GuidedFilter(frame_front, frame_front, 10, 0.001);
		mat_temp = frame_front.clone();

		cv::Mat front_gamma, back_gamma;
		if (mean(mat_temp)[0] > 128)
		{
			Adjust_gamma(mat_temp, front_gamma, 1.8);
		}
		else
		{
			mat_temp.copyTo(front_gamma);
		}
		if (mean(frame_back)[0] > 128)
		{
			Adjust_gamma(frame_back, back_gamma, 1.8);
		}
		else
		{
			frame_back.copyTo(back_gamma);
		}

		double dAlpha = 0.6;
		Laplace_pyramid_fusion(front_gamma, back_gamma, dAlpha, pyr_level, fusion_image);
		clock_t time_end = clock();
		std::cout << "lp_alpha_time:" << time_end - time_start << "ms" << std::endl;		
		std::cout << "lp all running time:" << time_end - time_1 << "ms" << std::endl;
		cout << "--------------------" << endl;
		//imshow("temp", front_gamma);

		vector<cv::Mat> vct_show_images;
		vct_show_images.push_back(frame_back);
		vct_show_images.push_back(frame_front);
		vct_show_images.push_back(frame_out);
		vct_show_images.push_back(fusion_image);
		cv::Mat allimage = All_Images_inWindow(vct_show_images);
		imshow("IR--VIS--ROI--Weight", allimage);

		//cv::Mat gamma;
		//if (mean(frame_out)[0] > 100)
		//{
		//	Adjust_gamma(frame_out, gamma, 1.8);
		//}
		//else
		//{
		//	//Adjust_gamma(frame_out, gamma);
		//	frame_out.copyTo(gamma);
		//}
		//imshow("gamma", gamma);


		//对比融合图像保存
		string str_img = string(IMAGES_PATH).append("fuse/");
		if (_access(str_img.c_str(), 0) == -1)	//如果文件夹不存在则创建
			auto xx = _mkdir(str_img.c_str());	
		//imwrite(str_img + to_string(i) + ".bmp", allimage);


		//融合图图像指标
		fusion_image_index k_fuse, w_fuse;
		image_fusion_evalution(frame_back, frame_front, frame_out, k_fuse);
		image_fusion_evalution(frame_back, frame_front, fusion_image, w_fuse);
		
		// //保存数据
		// //string file_path = getCurFilePath();//获取当前文件绝对路径
		// string pathname = "./FileData/";//相对路径
		// time_t t = time(0);
		// char ch[64];
		// //strftime(ch, sizeof(ch), "%Y-%m-%d-%H-%M-%S", localtime(&t)); //年-月-日 时-分-秒
		// strftime(ch, sizeof(ch), "%Y-%m-%d-%H-%M", localtime(&t)); //年-月-日 时-分
		// ofstream test_value(pathname + "k_fuse_and_w_fuse" + ch + ".txt", ios::app | ios::out);
		// test_value <<" k_fuse  " << "    w_fuse  " << endl;			
		// test_value <<k_fuse.fusion_EN << "	" << w_fuse.fusion_EN << std::endl;
		// test_value <<k_fuse.fusion_MI << "	" << w_fuse.fusion_MI << std::endl;
		// test_value <<k_fuse.fusion_meanGrad << "	" << w_fuse.fusion_meanGrad << std::endl;
		// test_value <<k_fuse.fusion_SSIM_irf << "	" << w_fuse.fusion_SSIM_irf << std::endl;
		// test_value <<k_fuse.fusion_SSIM_visf << "	" << w_fuse.fusion_SSIM_visf << std::endl;
		// test_value <<k_fuse.fusion_SF << "	" << w_fuse.fusion_SF << std::endl;
		// test_value <<k_fuse.fusion_SD << "	" << w_fuse.fusion_SD << std::endl;
		// test_value <<k_fuse.fusion_CC_irf << "	" << w_fuse.fusion_CC_irf << std::endl;
		// test_value <<k_fuse.fusion_CC_visf << "	" << w_fuse.fusion_CC_visf << std::endl;
		// //test_value <<k_fuse.fusion_VIF << "	" << w_fuse.fusion_VIF << std::endl;
		// //test_value <<k_fuse.fusion_Qab_f << "	" << w_fuse.fusion_Qab_f		<< std::endl;
		// test_value << "---------------------------------------------" << endl;
		// test_value.close();

		cv::waitKey();
	}

	cv::destroyAllWindows();
	return true;
}





