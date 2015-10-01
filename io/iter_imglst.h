/*
 * iter_imglst.h
 *
 *  Created on: Sep 27, 2015
 *      Author: florian
 */

#ifndef ITER_IMGLST_H_
#define ITER_IMGLST_H_

#include <stdio.h>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

#include "../utility/util.h"
#include "iter_base.h"
#include "image_augmenter.h"
#include "../utility/RNGen.h"
#include "scheduler.h"
#include <ctime>
#include "../neuralnet/configurator.h"


namespace iter{


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




class ImglstIter : public IIter< DataChunk > {

public:
  ImglstIter(std::string net, const bool & isTrain);

  /* Set parameters for ImglstIter */
  virtual void SetParam(const configVec & conf);

  /* Initalize ImglstIter */
  virtual void Initalize(void);

  /* Start ImglstIter */
  virtual void StartIter(const int & iter);

  /* Advance iterator */
  virtual bool Next(void);
  
  /* Return latest entry */
  virtual const DataChunk & Entry(void) const;
  
  /* Return latest entry */
  virtual DataChunk & Entry(void);

  virtual ~ImglstIter(void);

private:
  
  /* current data instance */
  DataChunk mDataBatch_;
  /* Shape of the dataChunk (numImg, channels, y, x) */
  mshadow::Shape<4> mShape_;
  /* total number of images in the lst */
  unsigned int mSize_;
  /* Maximum chunk size */
  unsigned int maxChunkSize_;
  /* Current chunk size */
  unsigned int mChunkSize_;
  
  /* is training set */
  bool isTrain_;
  
  /* Current iteration */
  int mIter_;
  /* Current read position */
  unsigned int mReadPos_;
  /* Path to imglst-file */
  std::string mPath_;
  /* List with img-paths */
  std::vector < std::pair < int, std::string > > mImglst_;

  /* Scheduler for class balance */
  Scheduler* mScheduler_;
  /* Augmenter to distort images */
  std::vector< cvimg::ImageAugmenter*  > mIAs_;
  cvimg::augparams augparameter_;

  //mean image
  mshadow::TensorContainer<cpu, 3> meanimg_;
  int nthread_;
  std::string net_;

private:
  
  /* Load image lst from mPath_*/
  void LoadImglist_(void);
  /* Load images and labels into dataChunk */
  void LoadImagesLabels_(const unsigned & size, const unsigned & tidsize, int tid);
};

ImglstIter::ImglstIter(std::string net, const bool & isTrain) :
		isTrain_(isTrain), mIter_(0), mReadPos_(0), mPath_(""), net_(net), mScheduler_(NULL), nthread_(4) {



}

ImglstIter::~ImglstIter(void) {
  delete mScheduler_;
  mScheduler_ = NULL;
  for(int t = 0; t <= nthread_; t++){
	  delete(mIAs_[t]);
	  mIAs_[t] = NULL;
  }
}

void ImglstIter::SetParam(const configVec & conf) {

	std::string trainpath;
	std::string testpath;

      /* Call setParam for scheduling */
      //mScheduler_->SetParam(conf); //HARDFUCK

    configVec conf2 = conf;
    
    configurator::readDataConfig(conf2, trainpath, testpath, augparameter_ );
    

    mPath_ = isTrain_ ? trainpath : testpath;

    mShape_[1] = augparameter_.net_input_shape_[0];	// toDo HARDFUCK
    mShape_[3] = augparameter_.net_input_shape_[2];
    mShape_[2] = augparameter_.net_input_shape_[1];
    maxChunkSize_ = configurator::getbatchsize(conf2);	// toDo read batchsize from conf


}

void ImglstIter::Initalize(void) {

  
  // initalize random-seed
  std::srand ( unsigned ( std::time(0) ) ); //unsigned ( std::time(0) )
  /* initalize scheduler for training */
  if ( isTrain_ ) mScheduler_ = new Scheduler();

  /* get number of threads */
  #pragma omp parallel
  {
	  nthread_ = std::max(omp_get_num_procs() / 2 - 1, 1);
  }
  /* initalize image augmenter */
  for(int t = 0; t <= nthread_; t++){
	  mIAs_.push_back(new cvimg::ImageAugmenter(augparameter_, isTrain_));
  }

  //Check whether image mean exists:
  bool image_mean_existend_ = utility::file_existent(net_ + "image_mean.csv");


	//Generate image mean
  	if(image_mean_existend_){
  		//Load image mean
  		std::cout << "Loading image mean from: " << net_ + "image_mean.csv" << std::endl;
  		meanimg_.Resize(Shape3(mShape_[1],mShape_[2],mShape_[3]));
  		load_tensor(meanimg_, net_ + "image_mean.csv");

  	}
  	else if(!image_mean_existend_ and isTrain_){
		std::cout << "Generating image mean at: " << net_ + "image_mean.csv" << std::endl;

		this->LoadImglist_();
		meanimg_.Resize(Shape3(mShape_[1],mShape_[2],mShape_[3]));
		meanimg_ = 0.0f;


		for(unsigned i = 0; i < mImglst_.size(); i++){
			mshadow::TensorContainer<cpu, 3> tmpimg;
			tmpimg.Resize(meanimg_.shape_);

			cv::Mat img = cv::imread( mImglst_[i].second, cv::IMREAD_COLOR );

			for(unsigned y = 0; y < tmpimg.size(1); ++y) {
			  for(unsigned x = 0; x < tmpimg.size(2); ++x) {
				cv::Vec3b bgr = img.at< cv::Vec3b >(y, x);
				// store in RGB order
				for(unsigned k = 0; k < mShape_[1]; k++){
					tmpimg[k][y][x] = (float)bgr[k];
				}
			  }
			}

			meanimg_ += tmpimg;

			if( (((i+1) % 10000 == 0) and (i != 0)) or ((i+1) == mImglst_.size())){
				std::cout << i+1 << " images processed" << std::endl;
			}


		}
		//norm mean img
		meanimg_ *= (1.0f / mImglst_.size());

		//Save image mean
		save_tensor(meanimg_, net_ + "image_mean.csv");

		mImglst_.clear();
		image_mean_existend_ = true;
		std::cout << std::endl;
	}
  	else{
  		utility::Error("Its highly recommended to use image mean from training to predict test set!");
  	}


}

void ImglstIter::StartIter(const int & iter) {
  
  /* update current iteration value*/
  mIter_ = iter;
  /* reset read position */
  mReadPos_ = 0;
  /* load image list */
  mImglst_.clear();
  /* load image list from file */
  this->LoadImglist_();
  /* If training use scheduler to balance class weighting */
  if ( isTrain_ ) mScheduler_->ApplySchedule(mImglst_, mIter_);
  /* Determine the upper limit for the chunkSize */
  mSize_ = mImglst_.size();
  mChunkSize_ = std::min(maxChunkSize_, mSize_);
  /* Random shuffle reading order */
  if ( isTrain_ ) std::random_shuffle ( mImglst_.begin(), mImglst_.end() );

  if(mIter_ == 0) std::cout << "current ChunkSize: " << mChunkSize_ << " max ChunkSize: " << maxChunkSize_ << " full size " << mSize_ << std::endl;
}

bool ImglstIter::Next(void) {
  
  if ( mReadPos_ < mSize_ ) {
    /* Determine maximum readable size */
    mShape_[0] = std::min(mChunkSize_, (mSize_ - mReadPos_));
    /* Resize data chunk if the size changed */
    if (mShape_[0] != mDataBatch_.Data.size(0)) mDataBatch_.Data.Resize(mShape_);
    mDataBatch_.Labels.resize(mShape_[0]);
    /* Load batch-images and batch-labels into mDataBatch_ */
    int tidsize = std::floor(static_cast<float>(mShape_[0]) / static_cast<float>(nthread_));
    #pragma omp parallel num_threads(nthread_)
    {
      int tid = omp_get_thread_num();
      this->LoadImagesLabels_(tidsize, tidsize, tid);
    }
    int rest = mShape_[0] - nthread_ * tidsize;
    if(rest != 0){
      this->LoadImagesLabels_(rest, tidsize, nthread_);
    }


    /* increment the current read position */
    mReadPos_ += mShape_[0];
    return true;
  }
  return false;
}


const DataChunk & ImglstIter::Entry(void) const {
  return mDataBatch_;
}

DataChunk & ImglstIter::Entry(void) {
  return mDataBatch_;
}

void ImglstIter::LoadImglist_(void) {
  std::ifstream dataSet (mPath_.c_str(), std::ios::in);
  if(!dataSet){ utility::Error("Data list file not found: %s", (char*)mPath_.c_str()); }

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
    mImglst_.push_back(tmp);
  }
  dataSet.close();
}

void ImglstIter::LoadImagesLabels_(const unsigned & size, const unsigned & tidsize, int tid){

  for( unsigned int i = 0; i < size; i++ ){
    /* load label */
    mDataBatch_.Labels[tid * tidsize + i] = (mImglst_[mReadPos_ + tid * tidsize + i].first);
    /* load image */
    cv::Mat img = cv::imread( mImglst_[mReadPos_ + tid * tidsize + i].second, cv::IMREAD_COLOR );

    if( false ) {
      cv::namedWindow( "pic" );
      cv::resize(img, img, cv::Size(512,512));
      cv::imshow( "pic", img );
      cv::waitKey(0);
    }

    /* distort images with opencv */
    img = mIAs_[tid]->distort(img);

    /* Copy data into mshadow tensor */
    for( unsigned int y = 0; y < mShape_[2]; ++y ) {
      for( unsigned int x = 0; x < mShape_[3]; ++x ) {
		cv::Vec3b bgr = img.at< cv::Vec3b >(y, x);
		/* store in RGB order */
		for( unsigned int k = 0; k < mShape_[1]; ++k ) {
		  mDataBatch_.Data[tid * tidsize + i][k][y][x] = ((float)bgr[k]);
		}
      }
    }

    //Substract image mean
    mDataBatch_.Data[tid * tidsize + i] -= meanimg_;

    img.release();


  }
  /* free open cv memory */
  
}


} // End iter-namespace

#endif /* ITER_IMGLST_H_ */
