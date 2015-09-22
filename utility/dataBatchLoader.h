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
#include "../utility/RNGen.h"

namespace dataload{

struct dataBatchLoader{

public:

	/*
	 * Create a 'dataBatchLoader' to read from 'lst_path' in 'batchSize' chunks
	 */
	dataBatchLoader(const unsigned int & batchSize, const bool & is_train, const bool & augmentData, std::vector < std::pair <std::string, std::string > > & cfg);

	/*
	 * Destructor
	 */
	virtual ~dataBatchLoader(void);

	/*
	 * Reads one databatch of size 'mJunkSize_'
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

	void start_epoch(unsigned epoch);

private:
	void Load_Images_Labels_(const unsigned &);
	void get_image_dims_();
	void load_data_list_();
	std::vector< real_t > schedule_current_weights(unsigned epoch);


private:

	unsigned int mPicSize_;					// picture-side length
	unsigned int mNumChannels_;
	unsigned int mJunkSize_;				// chunk-size
	unsigned int maxJunkSize_;
	unsigned int mReadCounter_;				// counter for number of batches
	unsigned int mReadPos_;					// current read position in the path-list
	unsigned int epoch_count_;              // count epochs
	bool is_train_;
	bool mAugmentData_;
	std::vector < std::pair <std::string, std::string > > cfg_;

	unsigned int mSize_;					// full data-size to process
	std::string mPath_;						// path to file
	bool mNumBatches__;						// state of the reader
	unsigned int mNumBatches;				// number of data-reads
	cvimg::ImageAugmenter* myIA_;                  // data structure to augment data
	cvimg::augparams augparameter_;

	std::vector < std::pair < int, std::string > > mImglst; // Train and test labels and datapath lists


	TensorContainer<cpu, 4, real_t> mImageData;		// Train and testdata container

	std::vector<int> mLabels;					// labels of the current batch

	RNGen* myRand_;
};

dataBatchLoader:: ~dataBatchLoader(void){
	delete(myIA_);
	delete(myRand_);
}

dataBatchLoader::dataBatchLoader(const unsigned int & junkSize, const bool & is_train, const bool & augmentData, std::vector < std::pair <std::string, std::string > > &cfg)
: mPicSize_(0), maxJunkSize_(junkSize), mJunkSize_(0), mNumBatches(0), mNumChannels_(0),
  mReadCounter_(0), mReadPos_(0), is_train_(is_train), mSize_(0), mPath_(""), mNumBatches__(false),
  mAugmentData_(augmentData), myIA_(NULL),
  cfg_(cfg),
  epoch_count_(0)
{

	std::string trainpath;
	std::string testpath;

	configurator::readDataConfig(cfg_,
							   trainpath,
							   testpath,
							   augparameter_ );

    mPath_ = is_train ? trainpath : testpath;

    // Set random seed
	std::srand ( 0 ); //unsigned ( std::time(0) )

	// Set picture side length
	this->get_image_dims_();

	if(mAugmentData_ and is_train_)	myIA_ = new cvimg::ImageAugmenter(augparameter_);

	myRand_ = new RNGen();
}

std::vector< real_t > dataBatchLoader::schedule_current_weights(unsigned epoch){
	std::vector< real_t > weights(augparameter_.weights_start.size());
	for(unsigned i = 0; i < weights.size(); i++){
		weights[i] = epoch * ((float)augparameter_.weights_end[i] - (float)augparameter_.weights_start[i]) / augparameter_.classweights_saturation_epoch + augparameter_.weights_start[i] ;
		if(epoch > augparameter_.classweights_saturation_epoch){
			weights[i] = augparameter_.weights_end[i];
		}
	}
	return weights;
}

void dataBatchLoader::start_epoch(unsigned epoch){

	epoch_count_ = epoch;

	// Read image-lists and determine complete datasize
	this->load_data_list_();


	//weight classes
	if(is_train_ and (augparameter_.weights_start.size() != 0)){
		std::vector< real_t > current_weights = schedule_current_weights(epoch_count_);
		unsigned size = mImglst.size();
		for(unsigned i = 0; i < size; i++){
			real_t prop = current_weights[mImglst[i].first] - 1;
			while(prop >= 1.0f){
				mImglst.push_back(mImglst[i]);
				prop = prop - 1.0f;
			}
			if(myRand_->bernoulli(prop)) mImglst.push_back(mImglst[i]);
		}
	}

	//Set sizes
	mSize_ = mImglst.size();
	mJunkSize_ = std::min(maxJunkSize_, mSize_);
	// Calculate number of data-batches
	mNumBatches = ceil(static_cast<float>(mSize_)/ static_cast<float>(mJunkSize_));

	if(epoch_count_ == 0) std::cout << "DataSize: " << mSize_ << " JunkSize: " << mJunkSize_ << std::endl;

	// Random is_train pathlist
	if (is_train_) std::random_shuffle ( mImglst.begin(), mImglst.end() );


}


void dataBatchLoader::readBatch(void) {

	// Only read next batch if neccessary
	if ( mReadPos_ < mSize_ ) {

		// Deterimine size of next batch
		unsigned int size = mSize_ - mReadPos_;	// Read data so far

		size =	std::min(mJunkSize_, size);

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
			myIA_->distort(img);
		}

		//substract image mean
		cv::Scalar avgPixelIntensity = myIA_->get_mean(img);

		for(unsigned y = 0; y < mImageData.size(2); ++y) {
		  for(unsigned x = 0; x < mImageData.size(3); ++x) {
			cv::Vec3b bgr = img.at< cv::Vec3b >(y, x);
			// store in RGB order
			for(unsigned k = 0; k < mNumChannels_; k++){
				mImageData[i][k][y][x] = ((float)bgr[k] - avgPixelIntensity[k]) / 256.0f; //toDo HARDCODE
			}
		  }
		}

	}

}


void dataBatchLoader::load_data_list_(){
      std::ifstream dataSet (mPath_.c_str(), std::ios::in);
      if(!dataSet){    	  utility::Error("Data list file not found: %s", (char*)mPath_.c_str());      }

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
      if(!dataSet){    	  utility::Error("Data list file not found: %s", (char*)mPath_.c_str());      }

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
	mImglst.clear();
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

}

#endif /* DATABATCHLOADER_H_ */
