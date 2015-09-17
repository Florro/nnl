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
#include "mshadow/tensor.h"

#define INTERPOL cv::INTER_CUBIC
#define BORDER cv::BORDER_CONSTANT

using namespace std;

class ImageAugmenter {
public:

	ImageAugmenter(){
		myRand = new RNGen();
	}

	void display_img(cv::Mat res){

		cv::resize(res,res,cv::Size(800,800));//resize image
	    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
	    cv::imshow( "Display window", res );                   // Show our image inside it.
	    cv::waitKey(0);                                          // Wait for a keystroke in the window

	}

	void distort(cv::Mat &src){

		cv::Mat rot_mat( 2, 3, CV_32FC1 );

		//MNIST
		/*
		bool fixrotate = false;
		bool mirror = false;
		int scaling = 1.25;
		int translation = 3;
		bool perspective = true;
		int rotationwarp = 15;
		real_t sheeringconstant = 0.05;


		//Plankton
		bool fixrotate = true;
		bool mirror = true;
		int scaling = 1.25;
		int translation = 7;
		bool perspective = true;
		int rotationwarp = 25;
		real_t sheeringconstant = 0.05;
		 */

		//Retina
		bool fixrotate = false;
		bool mirror = true;
		int scaling = 30;
		int translation = 5;
		bool perspective = true;
		int rotationwarp = 360; //other axis commented out
		real_t sheeringconstant = 0.00;


		//Randomly mirror on x/y axis, (no flip is 0.5 biased)
		if(mirror){
			if(myRand->bernoulli(0.5)){
			  cv::flip(src,src,myRand->bernoulli(0.5));
			}
		}

		//Randomly rotate 0°,90°,180°,270°
		if(fixrotate){

			real_t angle = 90*(int)myRand->uniform(0,4);
			real_t scale = 1.0;// + myRand->uniform(-0.15,0.15);

			cv::Point center = cv::Point( src.cols/2, src.rows/2 );

			/// Get the rotation matrix with the specifications above
			rot_mat = getRotationMatrix2D( center, angle, scale );

			/// Rotate the warped image
			cv::warpAffine( src, src, rot_mat, src.size(), INTERPOL, BORDER, cv::Scalar(0,0,0) );

		}

		if(perspective){
			rotateImage_(src, src, 90, 90, 90 + myRand->uniform(-rotationwarp,rotationwarp),
					0 + myRand->uniform(-translation,translation), 0 + myRand->uniform(-translation,translation), 200 + myRand->uniform(-scaling,scaling),200,
					myRand->uniform(-sheeringconstant,sheeringconstant), myRand->uniform(-sheeringconstant,sheeringconstant) );

		}


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

	void rotateImage_(const cv::Mat &input, cv::Mat &output, real_t alpha, real_t beta, real_t gamma, real_t dx, real_t dy, real_t dz, real_t f, real_t xs, real_t ys){

		//alpha: the rotation around the x axis
		//beta: the rotation around the y axis
		//gamma: the rotation around the z axis
		//dx: translation along the x axis
		//dy: translation along the y axis
		//dz: translation along the z axis (distance to the image)
		//f: focal distance
		//xs: fucking x sheer
		//ys: fucking y sheer

	    alpha = (alpha - 90.)*CV_PI/180.;
	    beta = (beta - 90.)*CV_PI/180.;
	    gamma = (gamma - 90.)*CV_PI/180.;

	    // get width and height for ease of use in matrices
	    real_t w = (real_t)input.cols;
	    real_t h = (real_t)input.rows;

	    // Projection 2D -> 3D matrix
	    cv::Mat A1 = (cv::Mat_<real_t>(4,3) <<
	              1, 0, -w/2,
	              0, 1, -h/2,
	              0, 0,    0,
	              0, 0,    1);

	    // Rotation matrices around the X, Y, and Z axis
	    cv::Mat RX = (cv::Mat_<real_t>(4, 4) <<
	              1,          0,           0, 0,
	              0, cos(alpha), -sin(alpha), 0,
	              0, sin(alpha),  cos(alpha), 0,
	              0,          0,           0, 1);

	    cv::Mat RY = (cv::Mat_<real_t>(4, 4) <<
	              cos(beta), 0, -sin(beta), 0,
	              0, 1,          0, 0,
	              sin(beta), 0,  cos(beta), 0,
	              0, 0,          0, 1);

	    cv::Mat RZ = (cv::Mat_<real_t>(4, 4) <<
	              cos(gamma), -sin(gamma), 0, 0,
	              sin(gamma),  cos(gamma), 0, 0,
	              0,          0,           1, 0,
	              0,          0,           0, 1);

	    // Composed rotation matrix with (RX, RY, RZ)
	    cv::Mat R = RX * RY * RZ;

	    // Translation matrix
	    cv::Mat T = (cv::Mat_<real_t>(4, 4) <<
	             1, 0, 0, dx,
	             0, 1, 0, dy,
	             0, 0, 1, dz,
	             0, 0, 0, 1);


	    cv::Mat S = (cv::Mat_<real_t>(4, 4) <<
				  1, xs, 0, 0,
				  ys, 1, 0, 0,
				  0, 0, 1, 0,
				  0, 0, 0, 1);

	    // 3D -> 2D matrix
	    cv::Mat A2 = (cv::Mat_<real_t>(3,4) <<
	              f, 0, w/2, 0,
	              0, f, h/2, 0,
	              0, 0,   1, 0);

	    // Final transformation matrix
	    cv::Mat trans = A2 * (T * S * (R * A1));

	    // Apply matrix transformation
	    cv::warpPerspective(input, output, trans, input.size(), INTERPOL, BORDER, cv::Scalar(0,0,0));

	 }


	mshadow::TensorContainer<cpu, 3> tmpres;
	RNGen* myRand;


};



#endif /* IMAGE_AUGMENTER_H_ */
