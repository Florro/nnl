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

	bool augment;
	bool mirror;
	int scaling;
	int translation;
	int rotation;
	int rotation_space;
	real_t sheering;
	std::string background;

	std::vector< real_t > weights_start;
	std::vector< real_t > weights_end;
	unsigned classweights_saturation_epoch;

};


class ImageAugmenter {
public:

	ImageAugmenter( augparams param):
						param_(param){

	}

	void display_img(cv::Mat res){

		cv::resize(res,res,cv::Size(800,800));//resize image
	    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    cv::imshow( "Display window", res );                   // Show our image inside it.
	    cv::waitKey(0);                                          // Wait for a keystroke in the window

	}

	cv::Mat distort(cv::Mat &src,  RNGen* myRand, bool augmentData){



		shape_ = Shape3(3,80,80);

		bool mirror_ = false;
		bool rand_crop_= 0;
		int rotate_ = -1.0f;
		int fill_value_ = 255;
		real_t max_aspect_ratio_ = 0.0;
		int max_rotate_angle_=0;
		real_t max_shear_ratio_=0;
		int min_crop_size_= -1;
		int max_crop_size_= -1;
		real_t min_random_scale_ = 1.0f;
		real_t max_random_scale_ = 1.0f;
		real_t min_img_size_ = 0.0f;
		real_t max_img_size_ = 1e10f;


		if(augmentData){
			mirror_ = true;
			rand_crop_= true;
			rotate_ = -1.0f;
			fill_value_ = 255;
			max_aspect_ratio_ = 0.5;
			max_rotate_angle_=180;
			max_shear_ratio_=0.5;
			min_crop_size_=80;
			max_crop_size_=80;
			min_random_scale_ = 0.4;
			max_random_scale_ = 1.8;
			min_img_size_ = 80;
			max_img_size_ = 136;
		}





		// shear
		float s = myRand->uniform(0,1) * max_shear_ratio_ * 2 - max_shear_ratio_;
		// rotate
		int angle = myRand->randint(max_rotate_angle_ * 2) - max_rotate_angle_;
		if (rotate_ > 0) angle = rotate_;
		if (rotate_list_.size() > 0) {
		  angle = rotate_list_[myRand->randint(rotate_list_.size() - 1)];
		}
		float a = cos(angle / 180.0 * M_PI);
		float b = sin(angle / 180.0 * M_PI);
		// scale
		float scale = myRand->uniform(0,1) * (max_random_scale_ - min_random_scale_) + min_random_scale_;
		// aspect ratio
		float ratio = myRand->uniform(0,1) * max_aspect_ratio_ * 2 - max_aspect_ratio_ + 1;
		float hs = 2 * scale / (1 + ratio);
		float ws = ratio * hs;
		// new width and height
		float new_width = std::max(min_img_size_, std::min(max_img_size_, scale * src.cols));
		float new_height = std::max(min_img_size_, std::min(max_img_size_, scale * src.rows));
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
						 cv::Scalar(fill_value_, fill_value_, fill_value_));
		cv::Mat res = temp;
		if (max_crop_size_ != -1 || min_crop_size_ != -1){
		  utils::Check(res.cols >= max_crop_size_ && res.rows >= max_crop_size_&&max_crop_size_ >= min_crop_size_,
			"input image size smaller than max_crop_size");
		  mshadow::index_t rand_crop_size = myRand->randint(max_crop_size_-min_crop_size_+1)+min_crop_size_;
		  mshadow::index_t y = res.rows - rand_crop_size;
		  mshadow::index_t x = res.cols - rand_crop_size;
		  if (rand_crop_ != 0) {
			y = myRand->randint(y + 1);
			x = myRand->randint(x + 1);
		  }
		  else {
			y /= 2; x /= 2;
		  }
		  cv::Rect roi(x, y, rand_crop_size, rand_crop_size);
		  cv::resize(res(roi), res, cv::Size(shape_[1], shape_[2]));
		}
		else{
		  utils::Check(static_cast<mshadow::index_t>(res.cols) >= shape_[1] && static_cast<mshadow::index_t>(res.rows) >= shape_[2],
			"input image size smaller than input shape");
		  mshadow::index_t y = res.rows - shape_[2];
		  mshadow::index_t x = res.cols - shape_[1];
		  if (rand_crop_ != 0) {
			y = myRand->randint(y + 1);
			x = myRand->randint(x + 1);
		  }
		  else {
			y /= 2; x /= 2;
		  }
		  cv::Rect roi(x, y, shape_[1], shape_[2]);
		  res = res(roi);
		}


		if(mirror_ && myRand->bernoulli(0.5)){
			cv::flip(res,res, myRand->bernoulli(0.5));
		}
		return res;


	}

	cv::Scalar get_mean(cv::Mat &image){

		//defines roi
		cv::Rect roi( 0, 0, image.size().width, image.size().height );
		//copies input image in roi
		cv::Mat image_roi = image( roi );
		//computes mean over roi
		cv::Scalar avgPixelIntensity = cv::mean( image_roi );

		return avgPixelIntensity;

	}






private:

	augparams param_;

	//helper
	std::vector<int> rotate_list_;
	// temporal space
	cv::Mat temp0, temp, temp2;
	/*! \brief input shape */
    mshadow::Shape<3> shape_;
};

}

#endif /* IMAGE_AUGMENTER_H_ */
