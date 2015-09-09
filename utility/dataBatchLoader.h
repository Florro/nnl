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
#include "util.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

struct dataBatchLoader{

public:

	/*
	 * Create a 'dataBatchLoader' to read from 'lst_path' in 'batchSize' chunks
	 */
	dataBatchLoader(const std::string & lst_path, const unsigned int & batchSize, const bool & shuffle);

	/*
	 * Destructor
	 */
	virtual ~dataBatchLoader(void) {};

	/*
	 * Reads one databatch of size 'mBatchSize'
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
	TensorContainer<cpu, 4, real_t> & X(void);

	/*
	 * Returns a reference to the lateste label-batch
	 */
	std::vector<int> & y(void);

	/*
	 * Returns 'true' if the complete dataset has been read, else 'false'
	 */
	const bool & finished(void) const;
	/*
	 * Returns the complete data-size to be processed
	 */
	const unsigned int & fullSize(void) const;

private:

	unsigned int mPicSize;					// picture-side length
	unsigned int mNumChannels;
	unsigned int mBatchSize;				// chunk-size
	unsigned int mReadCounter;				// counter for number of batches
	unsigned int mReadPos;					// current read position in the path-list
	bool mRandomShuffle;

	unsigned int mSize;						// full data-size to process
	std::string mPath;						// path to file
	bool mFinished;							// state of the reader
	unsigned int mNumBatches;				// number of data-reads


	std::vector< std::string > mImglst;		// Train and test data-path lists

	TensorContainer<cpu, 4, real_t> mX;		// Train and testdata container

	std::vector<int> mY;					// labels of the current batch
};

dataBatchLoader::dataBatchLoader(const std::string & lst_path, const unsigned int & batchSize, const bool & shuffle)
: mPicSize(128), mBatchSize(batchSize), mReadCounter(0), mReadPos(0), mRandomShuffle(shuffle), mSize(0), mPath(lst_path), mFinished(false)
{
	// Set random seed
	std::srand ( unsigned ( std::time(0) ) );

	// Set picture side length
	mPicSize = 96;
	mNumChannels = 1;

	// Read image-lists and determine complete datasize
	utility::load_data_list_retina_nolabels(lst_path.c_str(), mImglst);

	mSize = mImglst.size();
	mBatchSize = std::min(batchSize, mSize);

	// Calculate number of data-batches
	mNumBatches = ceil(static_cast<float>(mSize)/ static_cast<float>(mBatchSize));

	std::cout << "size: " << mSize << " batchsize " << mBatchSize << std::endl;
}

void dataBatchLoader::readBatch(void) {

	// Random shuffle pathlist
	if ( mReadCounter == 0 and mRandomShuffle) {
		std::random_shuffle ( mImglst.begin(), mImglst.end() );
	}

	// Only read next batch if neccessary
	if ( mReadPos < mSize ) {

		// Deterimine size of next batch
		unsigned int size = mSize - mReadPos;	// Read data so far

		size =	std::min(mBatchSize, size);

		// Resize data-container
		mX.Resize(Shape4(size, mNumChannels, mPicSize, mPicSize));
		// Resize label-container
		if ( mY.size() != 0) {
			mY.clear();
		}

		// Copy batch-labels into mY
		// Load batch-images into mX and batch-labels into mY
		//utility::LoadImages( mX, mImglst, mReadPos, size, 3);
		utility::Load_Images_Labels( mX, mY, mImglst, mReadPos, size, mNumChannels);

		// increment counters
		mReadCounter++;
		mReadPos += size;

		// Check if we are finished
		if ( mReadPos == mSize ) {
			mFinished = true;
		}
	}
}

void dataBatchLoader::reset(void) {
	mReadCounter = 0;
	mReadPos = 0;
	mFinished = false;
}

TensorContainer<cpu, 4, real_t> & dataBatchLoader::X(void) {
	return mX;
}

std::vector<int> & dataBatchLoader::y(void) {
	return mY;
}

const bool & dataBatchLoader::finished(void) const {
	return mFinished;
}

const unsigned int & dataBatchLoader::fullSize(void) const {
	return mSize;
}

const unsigned int & dataBatchLoader::numBatches(void) const {
	return mNumBatches;
}

#endif /* DATABATCHLOADER_H_ */
