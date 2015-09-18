/*
 * dataBatchLoader.h
 *
 *  Created on: Sep 5, 2015
 *      Author: florian
 */

#ifndef DATABATCHLOADER_H_
#define DATABATCHLOADER_H_

#include "mshadow/tensor.h"
#include "mshadow-ps/mshadow_ps.h"
#include "../utility/image_augmenter.h"
#include "util.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "../neuralnet/configurator.h"

struct dataBatchLoader{

public:

	/*
	 * Create a 'dataBatchLoader' to read from 'lst_path' in 'batchSize' chunks
	 */
	dataBatchLoader(const unsigned int & batchSize, const bool & shuffle, const bool & augmentData, std::vector < std::pair <std::string, std::string > > & cfg);

	/*
	 * Destructor
	 */
	virtual ~dataBatchLoader(void);

	/*
	 * Reads one databatch of size 'mBatchSize_'
	 */
	void readBatch(void);

	/*
	 * Resets all counters, next read will restart the batchprocess
	 */
	void reset(void);

	/*
	 * Returns the number of neccessary batch-passes to complete the reading process
	 */
	const unsigned int & numBatches(void) const;

	/*
	 * Returns a reference to the latest data-batch
	 */
	TensorContainer<cpu, 4, real_t> & Data(void);

	/*
	 * Returns a reference to the lateste label-batch
	 */
	std::vector<int> & Labels(void);

	/*
	 * Returns 'true' if the complete dataset has been read, else 'false'
	 */
	const bool & finished(void) const;
	/*
	 * Returns the complete data-size to be processed
	 */
	const unsigned int & fullSize(void) const;


private:
	void Load_Images_Labels_(const unsigned &);
	void get_image_dims_();
	void load_data_list_();


private:

	unsigned int mPicSize_;					// picture-side length
	unsigned int mNumChannels_;
	unsigned int mBatchSize_;				// chunk-size
	unsigned int mReadCounter_;				// counter for number of batches
	unsigned int mReadPos_;					// current read position in the path-list
	bool mRandomShuffle_;
	bool mAugmentData_;
	std::vector < std::pair <std::string, std::string > > cfg_;

	unsigned int mSize_;					// full data-size to process
	std::string mPath_;						// path to file
	bool mNumBatches__;						// state of the reader
	unsigned int mNumBatches;				// number of data-reads
	ImageAugmenter* myIA_;                  // data structure to augment data

	std::vector < std::pair < int, std::string > > mImglst; // Train and test labels and datapath lists


	TensorContainer<cpu, 4, real_t> mImageData;		// Train and testdata container

	std::vector<int> mLabels;					// labels of the current batch
};

dataBatchLoader:: ~dataBatchLoader(void){
	delete(myIA_);
}

dataBatchLoader::dataBatchLoader(const unsigned int & batchSize, const bool & shuffle, const bool & augmentData, std::vector < std::pair <std::string, std::string > > &cfg)
: mPicSize_(0), mBatchSize_(batchSize), mNumChannels_(0),
  mReadCounter_(0), mReadPos_(0), mRandomShuffle_(shuffle), mSize_(0), mPath_(""), mNumBatches__(false),
  mAugmentData_(augmentData), myIA_(NULL),
  cfg_(cfg)
{

	std::string trainpath;
	std::string testpath;
	bool mirror;
	int scaling;
	int translation;
	int rotation;
	real_t sheering;
	std::string background;

	readDataConfig(cfg_,
				   trainpath,
				   testpath,
				   mirror,
				   scaling,
				   translation,
				   rotation, //other axis commented out
				   sheering,
				   background );

    mPath_ = shuffle ? trainpath : testpath;

    std::cout << mPath_ << std::endl;
    std::cout << mirror << std::endl;
    std::cout << scaling << std::endl;
    std::cout << translation << std::endl;
    std::cout << rotation << std::endl;
    std::cout << sheering << std::endl;
    std::cout << background << std::endl;

	// Set random seed
	std::srand ( 0 ); //unsigned ( std::time(0) )

	// Set picture side length
	this->get_image_dims_();

	// Read image-lists and determine complete datasize
	this->load_data_list_();

	/*
	//equally weight classes
	if(shuffle){
		int size = mImglst.size();
		int weights[] = {0,10,5,15,20};
		for(int i = 0; i < size; i++){
			if(mImglst[i].first == 0){
				for(int j = 0; j < weights[0]; j++){
					mImglst.push_back(mImglst[i]);
				}
			}
			else if(mImglst[i].first == 1){
				for(int j = 0; j < weights[1]; j++){
					mImglst.push_back(mImglst[i]);
				}
			}
			else if(mImglst[i].first == 1){
				for(int j = 0; j < weights[2]; j++){
					mImglst.push_back(mImglst[i]);
				}
			}
			else if(mImglst[i].first == 1){
				for(int j = 0; j < weights[3]; j++){
					mImglst.push_back(mImglst[i]);
				}
			}
			else if(mImglst[i].first == 1){
				for(int j = 0; j < weights[4]; j++){
					mImglst.push_back(mImglst[i]);
				}
			}
		}
		for(int i = 0; i < 70; i++){
			mImglst.pop_back();
		}
	}

	*/

	if(mAugmentData_)	myIA_ = new ImageAugmenter(mirror,
												   scaling,
												   translation,
												   rotation, //other axis commented out
												   sheering,
												   background);

	mSize_ = mImglst.size();
	mBatchSize_ = std::min(batchSize, mSize_);

	// Calculate number of data-batches
	mNumBatches = ceil(static_cast<float>(mSize_)/ static_cast<float>(mBatchSize_));

	std::cout << "DataSize: " << mSize_ << " JunkSize: " << mBatchSize_ << std::endl;
}


void dataBatchLoader::readBatch(void) {

	// Random shuffle pathlist
	if ( mReadCounter_ == 0 and mRandomShuffle_) {
		std::random_shuffle ( mImglst.begin(), mImglst.end() );
	}

	// Only read next batch if neccessary
	if ( mReadPos_ < mSize_ ) {

		// Deterimine size of next batch
		unsigned int size = mSize_ - mReadPos_;	// Read data so far

		size =	std::min(mBatchSize_, size);

		// Resize data-container
		mImageData.Resize(Shape4(size, mNumChannels_, mPicSize_, mPicSize_));
		// Resize label-container
		if ( mLabels.size() != 0) {
			mLabels.clear();
		}

		// Load batch-images into mImageData and batch-labels into mLabels
		this->Load_Images_Labels_(size);

		// increment counters
		mReadCounter_++;
		mReadPos_ += size;

		// Check if we are finished
		if ( mReadPos_ == mSize_ ) {
			mNumBatches__ = true;
		}
	}
}

void dataBatchLoader::Load_Images_Labels_(const unsigned & size){

	for (unsigned i = 0; i < size; i++){

		//load label
		mLabels.push_back(mImglst[mReadPos_ + i].first);

		//load image
		cv::Mat img = cv::imread( mImglst[mReadPos_ + i].second, cv::IMREAD_COLOR );

		if(false){
			cv::namedWindow( "pic" );
			cv::resize(img, img, cv::Size(512,512));
			cv::imshow( "pic", img );
			cv::waitKey(0);
		}

		if(mAugmentData_){
			//distort image with opencv
			myIA_->display_img(img);
			myIA_->distort(img);
			myIA_->display_img(img);
		}

		//substract image mean
		cv::Scalar avgPixelIntensity = myIA_->get_mean(img);

		for(unsigned y = 0; y < mImageData.size(2); ++y) {
		  for(unsigned x = 0; x < mImageData.size(3); ++x) {
			cv::Vec3b bgr = img.at< cv::Vec3b >(y, x);
			// store in RGB order
			for(unsigned k = 0; k < mNumChannels_; k++){
				mImageData[i][k][y][x] = (float)bgr[k] - avgPixelIntensity[k]; //toDo HARDCODE
			}
		  }
		}

	}

}


void dataBatchLoader::load_data_list_(){
      std::ifstream dataSet (mPath_.c_str(), std::ios::in);
      if(!dataSet){    	  utility::Error("Data list file not found %s", mPath_.c_str());      }

      while (dataSet)
      {
        std::string s;
        if (!std::getline( dataSet, s )) break;
        std::pair < int, std::string > tmp;
        std::istringstream ss( s );
        int count = 0;
        while (ss)
        {
          std::string s;
          if (!getline( ss, s, ',' )) break;
          else if ( count % 2 == 0 ) tmp.first = atoi(s.c_str());
		  else if ( count % 2 == 1 ) tmp.second = s;
          count++;
        }
        mImglst.push_back(tmp);
      }
      dataSet.close();
}

void dataBatchLoader::get_image_dims_(){
      std::ifstream dataSet (mPath_.c_str(), std::ios::in);
      if(!dataSet){    	  utility::Error("Data list file not found %s", mPath_.c_str());      }

      std::string s;
      std::getline( dataSet, s );
      std::pair < int, std::string > tmp;
	  std::istringstream ss( s );
      int count = 0;
	  while (ss)
	  {
		  std::string s;
		  if (!getline( ss, s, ',' )) break;
		  else if ( count % 2 == 0 ) tmp.first = atoi(s.c_str());
		  else if ( count % 2 == 1 ) tmp.second = s;
		  count++;
	  }
      dataSet.close();

      cv::Mat img = cv::imread( (char*)tmp.second.c_str(), cv::IMREAD_COLOR );
      mPicSize_ = img.size().width;

      bool color = false;
      for(int i = 0; i < img.size().width; i++){
    	  for(int j = 0; j < img.size().height; j++){
    		  cv::Vec3b bgr = img.at< cv::Vec3b >(i, j);
			  if( (((float)bgr[0])  !=  ((float)bgr[1])) || (((float)bgr[0])  !=  ((float)bgr[1]))  || (((float)bgr[0])  !=  ((float)bgr[1])) ){
    			  color = true;
    		  }
    	  }
      }

      mNumChannels_ = color ? 3 : 1;

}

void dataBatchLoader::reset(void) {
	mReadCounter_ = 0;
	mReadPos_ = 0;
	mNumBatches__ = false;
}

TensorContainer<cpu, 4, real_t> & dataBatchLoader::Data(void) {
	return mImageData;
}

std::vector<int> & dataBatchLoader::Labels(void) {
	return mLabels;
}

const bool & dataBatchLoader::finished(void) const {
	return mNumBatches__;
}

const unsigned int & dataBatchLoader::fullSize(void) const {
	return mSize_;
}

const unsigned int & dataBatchLoader::numBatches(void) const {
	return mNumBatches;
}

#endif /* DATABATCHLOADER_H_ */
