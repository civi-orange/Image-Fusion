#include "Fusion_Image.h"

//#define IMAGES_PATH			""
//#define IMAGES_PATH			"../../AffineTrans/image/Tree_sequence/"
#define IMAGES_PATH		"../../AffineTrans/image/Duine_sequence/"
//#define IMAGES_PATH		"../../AffineTrans/image/Nato_camp_sequence/"
//#define IMAGES_PATH		"../../AffineTrans/image/city/"



//白天,过曝的情况，效果不好
int main()
{	
	
	
	// 初始化类方法，
	FIMG::method_group method_f;
	FIMG::Fusion_Image ffffffff;
	FIMG::Fusion_Image_Param pppp;
	pppp.dAlpha = 0.7;//融合比例设置
	ffffffff.init_parameter(pppp);

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
	method_f.listFiles(string(IMAGES_PATH).append("visual/"), vct_vis_files, "bmp");
	method_f.listFiles(string(IMAGES_PATH).append("thermal/"), vct_ir_files, "bmp");

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

		clock_t time_k1 = clock();
		cv::Mat target_target, mat_front, frame_out;
		mat_front = cv::Mat(front_gamma.size(), CV_8UC1, cv::Scalar(50));
		//抠图融合
		cv::Mat guit_mat = method_f.GuidedFilter(front_gamma, front_gamma, 100, 0.001);
		//imshow("guit_mat", guit_mat);
		method_f.Max_Entropy(guit_mat, target_target, 0, 0);
		//imshow("target_target", target_target);

		//红外目标区域融合微光图像
		cv::Mat test_mat_mat;
		back_gamma.copyTo(test_mat_mat);
		front_gamma.copyTo(test_mat_mat, target_target);
		imshow("target_fuse", test_mat_mat);
		clock_t time_k2 = clock();
		std::cout << "fuse_copy time:" << time_k2 - time_k1 << "ms" << std::endl;


		//cv::Mat fusion_image_k;
		//Laplace_pyramid_fusion(mat_front, frame_out, 0.8, 6, fusion_image_k);
		//Alpha_Beta_Image_fuse(mat_back, mat_front, 0.1, 0, fusion_image_k);//融合图像增加背景
		//add(frame_back, fusion_image_k, fusion_image_k);

		back_gamma.copyTo(frame_out);
		frame_front.copyTo(mat_front, target_target);
		cv::Mat fusion_image_k;
		ffffffff.fusion_image(mat_front, frame_out, fusion_image_k);
		fusion_image_k.copyTo(frame_out, target_target);
		clock_t time_k3 = clock();
		std::cout << "k time:" << time_k3 - time_k1 << "ms" << std::endl;
		//imshow("fusion_image_k", fusion_image_k);	
		



		//pyramid 权值融合图像------------------------------------------------------------------
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
		clock_t time_start = clock();		
		ffffffff.fusion_image(front_gamma, back_gamma, fusion_image);
		clock_t time_end = clock();
		std::cout << "lp_alpha_time:" << time_end - time_start << "ms" << std::endl;		
		std::cout << "lp all running time:" << time_end - time_1 << "ms" << std::endl;
		std::cout << "--------------------" << std::endl;
		//imshow("temp", front_gamma);


		//综合显示图像----------------------------------------------------------------
		vector<cv::Mat> vct_show_images;
		vct_show_images.push_back(frame_back);
		vct_show_images.push_back(frame_front);
		vct_show_images.push_back(frame_out);
		vct_show_images.push_back(fusion_image);
		cv::Mat allimage = method_f.All_Images_inWindow(vct_show_images);
		imshow("IR--VIS--ROI--Weight", allimage);


		//对比融合图像保存------------------------------------------------------
		string str_img = string(IMAGES_PATH).append("fuse/");
		if (_access(str_img.c_str(), 0) == -1)	//如果文件夹不存在则创建
			auto xx = _mkdir(str_img.c_str());	
		//imwrite(str_img + to_string(i) + ".bmp", allimage);

		//融合图图像指标
		FIMG::fusion_image_index k_fuse, w_fuse;
		ffffffff.Get_Fusion_indicator(frame_back, frame_front, frame_out, k_fuse);
		ffffffff.Get_Fusion_indicator(frame_back, frame_front, fusion_image, w_fuse);
		
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





