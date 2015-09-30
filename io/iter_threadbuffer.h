/*
 * iter_imglst.h
 *
 *  Created on: Sep 27, 2015
 *      Author: florian
 */

#ifndef ITER_THREADBUFFER_H_
#define ITER_THREADBUFFER_H_

#include "iter_base.h"
#include "thread_util.h"
#include "semaphore_util.h"

namespace iter{

class ThreadBufferIter : public IIter< DataChunk > {
  
public:
  
  ThreadBufferIter(IIter< DataChunk > * base);
  
    /* Set parameters for ThreadBufferIter */
  virtual void SetParam(const configVec & conf);

  /* Initalize ThreadBufferIter */
  virtual void Initalize(void);

  /* Start ThreadBufferIter */
  virtual void StartIter(const int & iter);

  /* Advance iterator */
  virtual bool Next(void);

  /* Return latest entry */
  virtual const DataChunk & Entry(void) const;
  
  /* Return latest entry */
  virtual DataChunk & Entry(void);

  virtual ~ThreadBufferIter(void);
  
private:
  
  /* Size of each buffer */
  static const unsigned int mBufferSize_ = 2;
  
  /* Base DataChunk iterator*/
  IIter< DataChunk > * mBaseIter_;
  
  /* Buffer one*/
  std::vector< DataChunk > mBufferA_;
  /* Buffer two */
  std::vector< DataChunk > mBufferB_;
  /* Current available DataChunk */
  DataChunk mBatch_;
  /* Size of Buffer A */
  unsigned int mEndA_;
  /* Size of Buffer B */
  unsigned int mEndB_;
  
  /* Buffer specifier */
  int mCurrentBuffer_;
  /* Idx for read access */
  unsigned int mBufferIdx_;
  /* Signal to terminate reading */
  bool mTerminateSignal_;
  /* initalized called */
  bool mIsInitalized_;
  
  /* Semaphore for load start */
  Semaphore mLoadNeed_;
  /* Semaphore for load end */
  Semaphore mLoadEnd_;
  
  /* New thread for the loading process */
  Thread mLoaderThread_;
  
private:
  /* Run the loader thread */
  inline void Run_(void);
  /* Launches the loading process */
  inline static void * LaunchLoader_(void* pthread);
  /* Get next DataChunk from the base iterator */
  inline bool Next_(DataChunk & buf);
  /* initalize the loader */
  inline void StartLoader_(void);
  /* switch the buffer */
  inline void SwitchBuffer_(void);
};

ThreadBufferIter::ThreadBufferIter(IIter< DataChunk > * base) 
: mBaseIter_(NULL), mEndA_(0), mEndB_(0), mCurrentBuffer_(0), mBufferIdx_(0), mTerminateSignal_(true), mIsInitalized_(false) {
  mBaseIter_ = base;
}

void ThreadBufferIter::SetParam(const configVec & conf) {

  /* Read parameters for Base iterator */
  mBaseIter_->SetParam(conf);

}

void ThreadBufferIter::Initalize(void) {
  /* Initalize Base iterator */
  mBaseIter_->Initalize();
  /* mallocate size for buffers */
  mBufferA_.resize(mBufferSize_);
  mBufferB_.resize(mBufferSize_);
}

void ThreadBufferIter::StartIter(const int & iter) {
  /* Only restart the loading process after the first iteration was performed */
  if ( mIsInitalized_ ) {
    /* Wait for the latest loading process to end */
    mLoadEnd_.Wait();
    /* Switch the buffer */
    mCurrentBuffer_ = 1;
    /* Restart Base iterator while thread buffer is waiting */
    mBaseIter_->StartIter(iter);
    /* Reset the buffer sizes */
    mEndA_ = mEndB_ = mBufferSize_;
    /* start the next buffer */
    mLoadNeed_.Post();
    /* Wait for the loader to finish */
    mLoadEnd_.Wait();
    /* Switch the buffer */
    mCurrentBuffer_ = 0;
    /* Start loading process */
    mLoadNeed_.Post();
    /* Reset the idx */
    mBufferIdx_ = 0;
  }
  else {
    /* Restart Base iterator */
    mBaseIter_->StartIter(iter);
    /* Start the loader thread */
    this->StartLoader_();
    mIsInitalized_ = true;
  }
}

bool ThreadBufferIter::Next(void) {
  if ( !mIsInitalized_ ) return false;
  if ( mBufferIdx_ == mBufferSize_ ) {
    /* switch to the second buffer */
    this->SwitchBuffer_();
    /* reset idx */
    mBufferIdx_ = 0;
  }
  /* no entry left in the buffer */
  if ( mBufferIdx_ >= (mCurrentBuffer_ ? mEndA_ : mEndB_) ) {
    return false;
  }
  /* Select the current buffer */
  std::vector< DataChunk > &buf = mCurrentBuffer_ ? mBufferA_ : mBufferB_;
  /* Copy DataChunk into the current available batch */
  mBatch_ = buf[mBufferIdx_];
  ++mBufferIdx_;
  return true;
}

const DataChunk & ThreadBufferIter::Entry(void) const {
  return mBatch_;
}

DataChunk & ThreadBufferIter::Entry(void) {
  return mBatch_;
}

ThreadBufferIter::~ThreadBufferIter(void) {
  /* Terminate the loading process */
  mTerminateSignal_ = true;
  mLoadNeed_.Post();
  /* Join the thread */
  if (mLoaderThread_.Join()) {
    fprintf(stderr, "Error joining thread\n");
  }
  /* Destroy semaphores */
  mLoadNeed_.Destroy();
  mLoadEnd_.Destroy();
  /* Free base iterator */
  delete mBaseIter_;
  mBaseIter_ = NULL;
  mIsInitalized_ = false;
}

inline void * ThreadBufferIter::LaunchLoader_(void* pthread) {
  static_cast< ThreadBufferIter* >(pthread)->Run_();
  Thread::Exit(NULL);
  return NULL;
}

inline bool ThreadBufferIter::Next_(DataChunk & buf) {
  if (mBaseIter_->Next()) {
    buf = mBaseIter_->Entry();
    return true;
  }
  return false;
}

inline void ThreadBufferIter::Run_(void) {
  /* Keep running while not terminated */
  while ( !mTerminateSignal_ ) {
    /* Lock this section */
    mLoadNeed_.Wait();
    /* Select the right buffer */
    std::vector< DataChunk > &buf = mCurrentBuffer_ ? mBufferB_ : mBufferA_;
    /* Load DataChunks into the current buffer */
    for ( unsigned int i = 0; i < mBufferSize_; ++i ) {
      /* Check if we could load the next DataChunk */
      if ( !Next_(buf[i]) ) {
	/* The complete Chunk could not be loaded */
	unsigned int & end = mCurrentBuffer_ ? mEndB_ : mEndA_;
	end = i;
	break;
      }
    }
    /* Mark End of loading process */
    mLoadEnd_.Post();   
  }
}

inline void ThreadBufferIter::StartLoader_(void) {
  /* Start loader run */
  mTerminateSignal_ = false;
  /* Set first buffer to load */
  mCurrentBuffer_ = 1;
  /* Load is needed */
  mLoadNeed_.Init(1);
  /* Load is not finished */
  mLoadEnd_.Init(0);
  /* Initalize buffer sizes to maximum */
  mEndA_ = mEndB_ = mBufferSize_;
  /* Start the new loader thread */
  mLoaderThread_.Start(LaunchLoader_, this);
  /* Wait for first load to finish */
  mLoadEnd_.Wait();
  /* Switch to the next buffer */
  mCurrentBuffer_ = 0;
  /* Start loading process */
  mLoadNeed_.Post();
  /* Reset buffer index */
  mBufferIdx_ = 0;  
}


inline void ThreadBufferIter::SwitchBuffer_(void) {
  /* wait for the loading process to end */
  mLoadEnd_.Wait();
  /* switch the buffer */
  mCurrentBuffer_ = !mCurrentBuffer_;
  /* Start loading process with new buffer */
  mLoadNeed_.Post();
}

} /* End iter namespace */

#endif /* ITER_THREADBUFFER_H_ */
