/*
 * image_augmenter.h
 *
 *  Created on: Aug 22, 2015
 *      Author: niklas
 */

#ifndef IMAGE_AUGMENTER_H_
#define IMAGE_AUGMENTER_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "RNGen.h"

namespace cvimg{

#define INTERPOL cv::INTER_CUBIC
#define BORDER cv::BORDER_CONSTANT

using namespace std;

struct augparams{

	bool mirror_;
	bool rand_crop_;
	int fill_value_;
	real_t max_aspect_ratio_;
	int max_rotate_angle_;
	real_t max_shear_ratio_;
	int min_crop_size_;
	int max_crop_size_;
	real_t min_random_scale_;
	real_t max_random_scale_;
	real_t min_img_size_;
	real_t max_img_size_;

	std::vector< real_t > weights_start;
	std::vector< real_t > weights_end;
	unsigned classweights_saturation_epoch;

	mshadow::Shape<3> net_input_shape_;

	augparams(){

		mirror_ = false;
		rand_crop_= false;
		fill_value_ = 255;
		max_aspect_ratio_ = 0.0;
		max_rotate_angle_=0;
		max_shear_ratio_=0;
		min_crop_size_= -1;
		max_crop_size_= -1;
		min_random_scale_ = 1.0f;
		max_random_scale_ = 1.0f;
		min_img_size_ = 0.0f;
		max_img_size_ = 1e10f;

		classweights_saturation_epoch = 1;

	}

	void reset(){ //HARDFUCK

		mirror_ = false;
		rand_crop_= false;
		fill_value_ = 255;
		max_aspect_ratio_ = 0.0;
		max_rotate_angle_=0;
		max_shear_ratio_=0;
		min_crop_size_= -1;
		max_crop_size_= -1;
		min_random_scale_ = 1.0f;
		max_random_scale_ = 1.0f;
		min_img_size_ = 0.0f;
		max_img_size_ = 1e10f;
		classweights_saturation_epoch = 1;

	}

};


class ImageAugmenter {
public:

	ImageAugmenter( augparams param, bool is_train ):
						param_(param){

		if(!is_train){
			param_.reset(); //HARDFUCK
		}

	}

	void display_img(cv::Mat res){

		cv::resize(res,res,cv::Size(800,800));//resize image
	    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    cv::imshow( "Display window", res );                   // Show our image inside it.
	    cv::waitKey(0);                                          // Wait for a keystroke in the window

	}

	cv::Mat distort(cv::Mat &src,  RNGen* myRand){


		/*
		std::cout << "mirror: " << param_.mirror_ << " 1" << std::endl;
		std::cout << "rand_crop: " << param_.rand_crop_ << " 1" << std::endl;
		std::cout << "fill_value: " << param_.fill_value_ << " 255"  << std::endl;
		std::cout << "max_aspect_ratio: " << param_.max_aspect_ratio_  << " 0.5" << std::endl;
		std::cout << "max_rotate_angle: " << param_.max_rotate_angle_  << " 180" << std::endl;
		std::cout << "max_shear_ratio: " << param_.max_shear_ratio_  << " 0.5" << std::endl;
		std::cout << "min_crop_size: " << param_.min_crop_size_  << " 80" << std::endl;
		std::cout << "max_crop_size: " << param_.max_crop_size_  << " 80" << std::endl;
		std::cout << "min_random_scale: " << param_.min_random_scale_  << " 0.4" << std::endl;
		std::cout << "max_random_scale: " << param_.max_random_scale_  << " 1.8" << std::endl;
		std::cout << "min_img_size: " << param_.min_img_size_  << " 80" << std::endl;
		std::cout << "max_img_size: " << param_.max_img_size_  << " 136" << std::endl;
		*/


		// shear
		float s = myRand->uniform(0,1) * param_.max_shear_ratio_ * 2 - param_.max_shear_ratio_;
		// rotate
		int angle = myRand->randint(param_.max_rotate_angle_ * 2) - param_.max_rotate_angle_;

		float a = cos(angle / 180.0 * M_PI);
		float b = sin(angle / 180.0 * M_PI);
		// scale
		float scale = myRand->uniform(0,1) * (param_.max_random_scale_ - param_.min_random_scale_) + param_.min_random_scale_;
		// aspect ratio
		float ratio = myRand->uniform(0,1) * param_.max_aspect_ratio_ * 2 - param_.max_aspect_ratio_ + 1;
		float hs = 2 * scale / (1 + ratio);
		float ws = ratio * hs;
		// new width and height
		float new_width = std::max(param_.min_img_size_, std::min(param_.max_img_size_, scale * src.cols));
		float new_height = std::max(param_.min_img_size_, std::min(param_.max_img_size_, scale * src.rows));
		//printf("%f %f %f %f %f %f %f %f %f\n", s, a, b, scale, ratio, hs, ws, new_width, new_height);
		cv::Mat M(2, 3, CV_32F);
		M.at<float>(0, 0) = hs * a - s * b * ws;
		M.at<float>(1, 0) = -b * ws;
		M.at<float>(0, 1) = hs * b + s * a * ws;
		M.at<float>(1, 1) = a * ws;
		float ori_center_width = M.at<float>(0, 0) * src.cols + M.at<float>(0, 1) * src.rows;
		float ori_center_height = M.at<float>(1, 0) * src.cols + M.at<float>(1, 1) * src.rows;
		M.at<float>(0, 2) = (new_width - ori_center_width) / 2;
		M.at<float>(1, 2) = (new_height - ori_center_height) / 2;
		cv::warpAffine(src, temp, M, cv::Size(new_width, new_height),
						 cv::INTER_LINEAR,
						 cv::BORDER_CONSTANT,
						 cv::Scalar(param_.fill_value_, param_.fill_value_, param_.fill_value_));
		cv::Mat res = temp;
		if (param_.max_crop_size_ != -1 || param_.min_crop_size_ != -1){
		  utility::Check(res.cols >= param_.max_crop_size_ && res.rows >= param_.max_crop_size_&&param_.max_crop_size_ >= param_.min_crop_size_,
			"input image size smaller than max_crop_size");
		  mshadow::index_t rand_crop_size = myRand->randint(param_.max_crop_size_-param_.min_crop_size_+1)+param_.min_crop_size_;
		  mshadow::index_t y = res.rows - rand_crop_size;
		  mshadow::index_t x = res.cols - rand_crop_size;
		  if (param_.rand_crop_ != 0) {
			y = myRand->randint(y + 1);
			x = myRand->randint(x + 1);
		  }
		  else {
			y /= 2; x /= 2;
		  }
		  cv::Rect roi(x, y, rand_crop_size, rand_crop_size);
		  cv::resize(res(roi), res, cv::Size(param_.net_input_shape_[1], param_.net_input_shape_[2]));
		}
		else{
		  utility::Check(static_cast<mshadow::index_t>(res.cols) >= param_.net_input_shape_[1] && static_cast<mshadow::index_t>(res.rows) >= param_.net_input_shape_[2],
			"input image size smaller than input shape");
		  mshadow::index_t y = res.rows - param_.net_input_shape_[2];
		  mshadow::index_t x = res.cols - param_.net_input_shape_[1];
		  if (param_.rand_crop_ != 0) {
			y = myRand->randint(y + 1);
			x = myRand->randint(x + 1);
		  }
		  else {
			y /= 2; x /= 2;
		  }
		  cv::Rect roi(x, y, param_.net_input_shape_[1], param_.net_input_shape_[2]);
		  res = res(roi);
		}


		if(param_.mirror_ && myRand->bernoulli(0.5)){
			cv::flip(res,res, myRand->bernoulli(0.5));
		}
		return res;


	}



private:

	augparams param_;
	// temporal space
	cv::Mat temp;

};

}

#endif /* IMAGE_AUGMENTER_H_ */
