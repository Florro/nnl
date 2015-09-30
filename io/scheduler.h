/*
 * schduler.h
 *
 *  Created on: Sep 27, 2015
 *      Author: florian
 */

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <string>

#include "../utility/RNGen.h"

typedef std::vector < std::pair <std::string, std::string > > configVec;

class Scheduler {
  
public:

  Scheduler(void);
  
  /* Set parameters for Scheduler */
  void SetParam(const configVec & conf);  
  /* apply the current weights to the img lst */
  void ApplySchedule(std::vector < std::pair < int, std::string > > & imglst, const int & iter);
  
  virtual ~Scheduler(void);

private:
  
  /* Current iteration */
  int mIter_;
  /* iteration to saturate the scheduling */
  int saturationIter_;

  /* schedul weights for all classes*/
  std::vector< float > startWeights_;
  std::vector< float > endWeights_;
  std::vector< float > weights_;
  
  /* Random number generator for scheduling */
  RNGen* myRand_;
  
  
private:
  /* update the weights for the next iteration */
  void UpdateSchedule_(void);
  
};

Scheduler::Scheduler() : mIter_(0), saturationIter_(0) {
  myRand_ = NULL;
}

void Scheduler::SetParam(const configVec & conf) {
  for( unsigned int i = 0; i < conf.size(); ++i) {
    if (!strcmp(conf[i].first.c_str(), "classweights_saturation_epoch")) saturationIter_ = atoi(conf[i].second.c_str());
  }
}

Scheduler::~Scheduler(void) {
  delete myRand_;
  myRand_ = NULL;
}

void Scheduler::ApplySchedule(std::vector < std::pair < int, std::string > > & imglst, const int & iter) {
  
  mIter_ = iter;
  
  if( startWeights_.size() != 0 ){
    // Calculate the new weighting based on the current iteration
    this->UpdateSchedule_();
    unsigned int size = imglst.size();
    for(unsigned int i = 0; i < size; i++){
      float prop = weights_[imglst[i].first] - 1;
      while(prop >= 1.0f){
	imglst.push_back(imglst[i]);
	prop = prop - 1.0f;
      }
      if(myRand_->bernoulli(prop)) imglst.push_back(imglst[i]);
    }
  }
  
}

void Scheduler::UpdateSchedule_(void) {
  
  weights_.resize(startWeights_.size());
  for(unsigned int i = 0; i < weights_.size(); i++){
    weights_[i] = mIter_ * ((float)endWeights_[i] - (float)startWeights_[i]) / saturationIter_ + startWeights_[i] ;
    if(mIter_ > saturationIter_){
      weights_[i] = endWeights_[i];
    }
  }
}



#endif /* SCHEDULER_H_ */
