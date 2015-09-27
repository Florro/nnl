/*
 * dataBatchLoader.h
 *
 *  Created on: Sep 5, 2015
 *      Author: florian
 */

#ifndef DATABATCHLOADER_MTHREAD_H_
#define DATABATCHLOADER_MTHREAD_H_

#include "mshadow/tensor.h"
#include "mshadow-ps/mshadow_ps.h"
#include "../utility/image_augmenter2.h"
#include "util.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "../neuralnet/configurator.h"
#include "../utility/RNGen.h"
#include <omp.h>


namespace dataload{

//Write params into file
void save_tensor(TensorContainer<cpu, 3, real_t> host_weight, std::string outputfile){

  std::ofstream outputstream ((char*)outputfile.c_str());

  if (outputstream.is_open()){
	  for(unsigned i = 0; i < host_weight.size(0); i++){
		  for(unsigned j = 0; j < host_weight.size(1); j++){
			  for(unsigned k = 0; k < host_weight.size(2); k++){
				  outputstream << host_weight[i][j][k] << ",";
			  }
			  outputstream << std::endl;
		  }
		  outputstream << std::endl << std::endl;
	  }
  outputstream.close();
  }
  else{
	  utility::Error("Saving image mean failed");
  }

}


void load_tensor(TensorContainer<cpu, 3, real_t> &host_weight, std::string inputfile){

	  std::ifstream dataSet ((char*)inputfile.c_str());
	  if(!dataSet){
		  utility::Error("Loading image mean failed");
	  }

	  unsigned globalcount = 0;
	  unsigned count1 = 0;
	  unsigned count2 = 0;
	  unsigned count3 = 0;

	  while (dataSet)
	  {
		  std::string s;
		  if (!std::getline( dataSet, s )) break;
		  std::istringstream ss( s );
		  while (ss)
		  {
			std::string s;
			if (!getline( ss, s, ',' )) break;

			count3 = globalcount % host_weight.size(2);
			count2 = (globalcount / host_weight.size(2)) % host_weight.size(1);
			count1 = (globalcount / ( host_weight.size(2) * host_weight.size(1) )) % host_weight.size(0);

			host_weight[count1][count2][count3] = atof(s.c_str());

			globalcount++;
		  }
	  }
	  dataSet.close();


}




struct dataBatchLoader_mthread{

public:

	/*
	 * Create a 'dataBatchLoader' to read from 'lst_path' in 'batchSize' chunks
	 */
	dataBatchLoader_mthread(const unsigned int & batchSize, const bool & is_train, const bool & augmentData, std::vector < std::pair <std::string, std::string > > & cfg);

	/*
	 * Destructor
	 */
	virtual ~dataBatchLoader_mthread(void);

	/*
	 * Reads one databatch of size 'mchunkSize_'
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
	void Load_Images_Labels_(const unsigned & , const unsigned &, int tid);
	void get_image_dims_();
	void load_data_list_();
	std::vector< real_t > schedule_current_weights(unsigned epoch);


private:

	unsigned int mPicSize_;					// picture-side length
	unsigned int mNumChannels_;
	unsigned int mchunkSize_;				// chunk-size
	unsigned int maxchunkSize_;
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

	cvimg::augparams augparameter_;

	std::vector < std::pair < int, std::string > > mImglst; // Train and test labels and datapath lists


	TensorContainer<cpu, 4, real_t> mImageData;		// Train and testdata container

	std::vector<int> mLabels;					// labels of the current batch

	mshadow::TensorContainer<cpu, 3> meanimg_;

	std::vector<RNGen*> myRands_;
	std::vector<cvimg::ImageAugmenter* > myIAs_;               // data structure to augment data

	int nthread_;
};

dataBatchLoader_mthread:: ~dataBatchLoader_mthread(void){


	for(int i = 0; i < nthread_; i++){
		delete(myIAs_[i]);
		delete(myRands_[i]);
	}


}

dataBatchLoader_mthread::dataBatchLoader_mthread(const unsigned int & chunkSize, const bool & is_train, const bool & augmentData, std::vector < std::pair <std::string, std::string > > &cfg)
: mPicSize_(0), maxchunkSize_(chunkSize), mchunkSize_(0), mNumBatches(0), mNumChannels_(0),
  mReadCounter_(0), mReadPos_(0), is_train_(is_train), mSize_(0), mPath_(""), mNumBatches__(false),
  mAugmentData_(augmentData),
  cfg_(cfg),
  epoch_count_(0),
  nthread_(4)
{

	std::string trainpath;
	std::string testpath;

	configurator::readDataConfig(cfg_,
							   trainpath,
							   testpath,
							   augparameter_ );

    mPath_ = is_train ? trainpath : testpath;

    // Set random seed
	std::srand ( unsigned ( std::time(0) ) );

	// Set picture side length
	this->get_image_dims_();

	#pragma omp parallel
	{
		nthread_ = std::max(omp_get_num_procs() / 2 - 1, 1);
	}
	for(int i = 0; i < nthread_; i++){
		myIAs_.push_back(new cvimg::ImageAugmenter(augparameter_));
	}
	for(int i = 0; i < nthread_; i++){
		myRands_.push_back(new RNGen());
	}

}

std::vector< real_t > dataBatchLoader_mthread::schedule_current_weights(unsigned epoch){
	std::vector< real_t > weights(augparameter_.weights_start.size());
	for(unsigned i = 0; i < weights.size(); i++){
		weights[i] = epoch * ((float)augparameter_.weights_end[i] - (float)augparameter_.weights_start[i]) / augparameter_.classweights_saturation_epoch + augparameter_.weights_start[i] ;
		if(epoch > augparameter_.classweights_saturation_epoch){
			weights[i] = augparameter_.weights_end[i];
		}
	}
	return weights;
}

void dataBatchLoader_mthread::start_epoch(unsigned epoch){

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
			if(myRands_[0]->bernoulli(prop)) mImglst.push_back(mImglst[i]);
		}
	}

	//Set sizes
	mSize_ = mImglst.size();
	mchunkSize_ = std::min(maxchunkSize_, mSize_);
	// Calculate number of data-batches
	mNumBatches = ceil(static_cast<float>(mSize_)/ static_cast<float>(mchunkSize_));

	if(epoch_count_ == 0) std::cout << "DataSize: " << mSize_ << " chunkSize: " << mchunkSize_ << std::endl;

	// Random is_train pathlist
	if (is_train_) std::random_shuffle ( mImglst.begin(), mImglst.end() );

	meanimg_.Resize(Shape3(3,80,80));//HARDFUCK
	std::string name_meanimg_ = "/home/niklas/CXX/nnl/testNets/plankton/cxx_compare2/image_mean"; //HARDFUCK
	load_tensor(meanimg_, name_meanimg_); //HARDFUCK

}


void dataBatchLoader_mthread::readBatch(void) {

	// Only read next batch if neccessary
	if ( mReadPos_ < mSize_ ) {

		// Deterimine size of next batch
		unsigned int size = mSize_ - mReadPos_;	// Read data so far

		size =	std::min(mchunkSize_, size);

		// Resize data-container
		mImageData.Resize(Shape4(size, mNumChannels_, mPicSize_, mPicSize_));
		// Resize label-container
		if ( mLabels.size() != 0) {
			mLabels.clear();
		}
		mLabels.resize(size);

		// Load batch-images into mImageData and batch-labels into mLabels
		int tidsize = std::floor(static_cast<float>(size) / static_cast<float>(nthread_));
		#pragma omp parallel num_threads(nthread_)
		{
			int tid = omp_get_thread_num();
			this->Load_Images_Labels_(tidsize, tidsize, tid);
		}
		int rest = size - nthread_ * tidsize;
		if(rest != 0){
			this->Load_Images_Labels_(rest, tidsize, nthread_);
		}

		// increment counters
		mReadCounter_++;
		mReadPos_ += size;

		// Check if we are finished
		if ( mReadPos_ == mSize_ ) {
			mNumBatches__ = true;
		}
	}
}



void dataBatchLoader_mthread::Load_Images_Labels_(const unsigned & size, const unsigned & tidsize, int tid){

	for (unsigned i = 0; i < size; i++){

		//load label
		mLabels[tid * tidsize + i] = (mImglst[mReadPos_ + tid * tidsize + i].first);

		//load image
		cv::Mat img = cv::imread( mImglst[mReadPos_ + tid * tidsize + i].second, cv::IMREAD_COLOR );

		if(false){
			cv::namedWindow( "pic" );
			cv::resize(img, img, cv::Size(512,512));
			cv::imshow( "pic", img );
			cv::waitKey(0);
		}


		//distort image with opencv
		img = myIAs_[tid]->distort(img, myRands_[tid], mAugmentData_);

		//substract image mean
		//cv::Scalar avgPixelIntensity = myIA_->get_mean(img);

		for(unsigned y = 0; y < mImageData.size(2); ++y) {
		  for(unsigned x = 0; x < mImageData.size(3); ++x) {
			cv::Vec3b bgr = img.at< cv::Vec3b >(y, x);
			// store in RGB order
			for(unsigned k = 0; k < mNumChannels_; k++){
				//mImageData[tid * tidsize + i][k][y][x] = ((float)bgr[k] - avgPixelIntensity[k]); //toDo HARDCODE
				mImageData[tid * tidsize + i][k][y][x] = (float)bgr[k];
			}
		  }
		}

		//meanimg_ += mImageData[i]; //HARDFUCK //just single thread!!!
		mImageData[tid * tidsize + i] -= meanimg_; //HARDFUCK

	}


	//std::string name_meanimg_ = "/home/niklas/CXX/nnl/testNets/plankton/cxx_compare2/image_mean"; //HARDFUCK
	//meanimg_ /= 20000.0f; //HARDFUCK
	//save_tensor(meanimg_, name_meanimg_); //HARDFUCK


}


void dataBatchLoader_mthread::load_data_list_(){
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

void dataBatchLoader_mthread::get_image_dims_(){
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
      cv::resize(img,img,cv::Size(80,80)); //HARDFUCK
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
      mNumChannels_ = 3; // HARDFUCK
}

void dataBatchLoader_mthread::reset(void) {
	mReadCounter_ = 0;
	mReadPos_ = 0;
	mNumBatches__ = false;
	mImglst.clear();
}

TensorContainer<cpu, 4, real_t> & dataBatchLoader_mthread::Data(void) {
	return mImageData;
}

std::vector<int> & dataBatchLoader_mthread::Labels(void) {
	return mLabels;
}

const bool & dataBatchLoader_mthread::finished(void) const {
	return mNumBatches__;
}

const unsigned int & dataBatchLoader_mthread::fullSize(void) const {
	return mSize_;
}

const unsigned int & dataBatchLoader_mthread::numBatches(void) const {
	return mNumBatches;
}







}

#endif /* DATABATCHLOADER_H_ */
