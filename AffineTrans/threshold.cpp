#include "funcHead.h"

#define IMAGES_PATH		"D:/WorkData/dataset/Terravic/irw01/"																//.jpg
//#define IMAGES_PATH		"D:/WorkData/dataset/osu/00007/"																	//.bmp
//#define IMAGES_PATH		"D:/WorkData/dataset/TNO_Image_Fusion_Dataset/FEL_images/Nato_camp_sequence/thermal/"				//.bmp
//#define IMAGES_PATH		"D:/WorkData/dataset/TNO_Image_Fusion_Dataset/FEL_images/Duine_sequence/thermal/"					//.bmp
//#define IMAGES_PATH		"D:/WorkData/dataset/TNO_Image_Fusion_Dataset/DHV_images/Fire_sequence/part_1/thermal/"				//.bmp
//#define IMAGES_PATH		"D:/WorkData/dataset/TNO_Image_Fusion_Dataset/FEL_images/Tree_sequence/thermal/"					//.bmp


int main_test()
{
	vector<cv::String> vct_files;
	listFiles(IMAGES_PATH, vct_files, "jpg");

	if (vct_files.size() == 0)
	{

		cv::String vis = "./image/two/soldier1/VIS_meting012-1200_r.bmp";
		vct_files.push_back(vis);
		cv::Mat frame = cv::imread("./image/two/soldier1/meting012-1200_rg.bmp");
		cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		imshow("what", frame);
		cv::waitKey();
	}

	string pathname = "./image_info/";//相对路径
	time_t t = time(0);
	char ch[64];
	strftime(ch, sizeof(ch), "%Y-%m-%d-%H-%M", localtime(&t)); //年-月-日 时-分
	ofstream txt_info(pathname + ch + ".txt", ios::app | ios::out);

	string path = IMAGES_PATH;

	
	for (int i = 0; i < vct_files.size(); i++)
	{
		//获取图像
		cv::Mat frame;
		//cv::VideoCapture cap;
		//cap.open("");
		frame = cv::imread(vct_files[i]);
		cv::Mat gray;
		cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		//计算图像信息
		float fEN = entropy_a(gray);
		float fEN2 = entropy_2(gray);
		float fSF = calc_SF(gray);
		float fMG = mean_grad(gray);
		cv::Mat mean_f, std_f;
		meanStdDev(gray, mean_f, std_f);
		float fSD = std_f.at<double>(0, 0);

		txt_info << fEN << std::endl;
		txt_info << fEN2 << std::endl;
		txt_info << fSF << std::endl;
		txt_info << fMG << std::endl;
		txt_info << fSD << std::endl;
		txt_info << "---------------------------------------------" << endl;
		putText(gray, to_string(fEN).append(" / ").append(to_string(fEN2)), cv::Point(10, 10), 0, 0.3, cv::Scalar(255));
		putText(gray, to_string(fSF).append(" / ").append(to_string(fMG)), cv::Point(10, 20), 0, 0.3, cv::Scalar(255));
		putText(gray, to_string(fSD), cv::Point(10, 30), 0, 0.3, cv::Scalar(255));
		imshow("g", gray);
		cv::waitKey(1);
		//imshow("gray", gray);
		//resize(gray, gray, cv::Size(640, 512), 0, 0, 3);
		// clock_t t3 = clock();
		// cv::Mat mat_out;
		// Get_Infrared_target_region(gray, mat_out);
		// clock_t t4 = clock();
		// cout << "run_time:" << t4 - t3 << endl;
		// cout << "--------------------" << endl;
		// 
		// clock_t t00 = clock();
		// cv::Mat guid_out = GuidedFilter(gray, gray, 10, 0.01);
		// clock_t t000 = clock();
		// cout << "guid:" << t000 - t00 << endl;
		// imshow("guid_out", guid_out);
		// 
		// imshow("big", mat_out);
		// cv::waitKey(1);
	}



		//Py_SetPythonHome(L"D:/Anaconda3/");
		//Py_Initialize();
		////if (!Py_IsInitialized()) {
		////	return -1;
		////}
		//import_array();
		////PyRun_SimpleString("import os");
		////PyRun_SimpleString("print(os.listdir())");
		//PyRun_SimpleString("import sys");
		//PyRun_SimpleString("sys.path.append('./')");
		//PyObject* pModule = PyImport_ImportModule("python_code");
		//
		////PyObject* pDict = PyModule_GetDict(pModule);
		////PyObject* pFunc = PyDict_GetItemString(pDict, "vifp_mscale");
		//PyObject* pFunc = PyObject_GetAttrString(pModule, "matfunc");
		//PyObject* pFunc_add = PyObject_GetAttrString(pModule, "addfunc");
		//if (!pFunc || !PyCallable_Check(pFunc))
		//{
		//	cout << "Can't find vifp_mscale" << endl;
		//}
		//if (!pFunc_add || !PyCallable_Check(pFunc_add))
		//{
		//	cout << "Can't find _addab" << endl;
		//}
		//
		//int r = frame.rows;
		//int c = frame.cols;
		//int chnl = frame.channels();
		//// total number of elements (here it's an RGB image of size 640x480)
		//int nElem = r * c * chnl;
		//// the dimensions of the matrix
		//npy_intp mdim[] = { r, c, chnl };
		//// create an array of apropriate datatype
		//uchar* m1 = new uchar[nElem];
		//uchar* m2 = new uchar[nElem];
		//// copy the data from the cv::Mat object into the array
		//std::memcpy(m1, frame.data, nElem * sizeof(uchar));
		//std::memcpy(m2, mat_out.data, nElem * sizeof(uchar));
		//
		//// convert the cv::Mat to numpy.array NPY_INT32,NPY_UINT8
		//PyObject* mat1 = PyArray_SimpleNewFromData(chnl, mdim, NPY_UINT8, (void*)m1);
		//PyObject* mat2 = PyArray_SimpleNewFromData(chnl, mdim, NPY_UINT8, (void*)m2);
		//// create a Python-tuple of arguments for the function call
		//// "()" means "tuple". "O" means "object"
		////PyObject* args1 = Py_BuildValue("(O)", mat1);
		
		//PyObject* pArgs = PyTuple_New(2);
		//PyTuple_SetItem(pArgs, 0, Py_BuildValue("(O)", mat1));
		//PyTuple_SetItem(pArgs, 1, Py_BuildValue("(O)", mat2));
		
		//PyObject* pArgsab = PyTuple_New(2);
		//PyTuple_SetItem(pArgsab, 0, PyLong_FromLong(3));
		//PyTuple_SetItem(pArgsab, 1, PyLong_FromLong(4));
		//PyObject* pReturn1 = PyEval_CallObject(pFunc_add, pArgsab);
		//// process the result
		//float result1 = 0;
		//PyArg_Parse(pReturn1, "f", &result1);
		//cout << "add1:" << result1 << endl;
		
		//// execute the function
		//PyObject* pReturn = PyEval_CallObject(pFunc, pArgs);
		//// process the result
		//float result = 0;
		//PyArg_Parse(pReturn, "f", &result);
		//cout << "image:" << result << endl;
		
		//// decrement the object references
		
		//Py_XDECREF(mat1);
		//Py_XDECREF(mat2);
		//Py_XDECREF(pReturn1);
		//Py_XDECREF(pReturn);
		//Py_XDECREF(pArgs);
		//Py_XDECREF(pModule);
		//delete[] m1;
		//delete[] m2;	
	//Py_Finalize();

	txt_info.close();
	cv::destroyAllWindows();
	return 1;
}