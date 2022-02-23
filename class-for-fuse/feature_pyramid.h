#pragma once
#include "parameter.h"

namespace FIMG
{ 
class feature_pyramid
{
public:

	feature_pyramid();

	virtual ~feature_pyramid();

	int SetParam(const Fusion_Image_Param& param);

	int Gauss_Pyr(const cv::Mat& input, Pyr_FEAT& Img_pyr);

	int Laplace_Pyr(Pyr_FEAT& Img_Gaussian_pyr, Pyr_FEAT& Img_Laplacian_pyr);


private:

	Fusion_Image_Param param_;

};

}
