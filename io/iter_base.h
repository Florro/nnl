/*
 * iter_base.h
 *
 *  Created on: Sep 27, 2015
 *      Author: florian
 */

#ifndef ITER_BASE_H_
#define ITER_BASE_H_

#include <mshadow/tensor.h>

typedef std::vector < std::pair <std::string, std::string > > configVec;

namespace iter{

template< typename DataType > class IIter {
public:
  /* Set parameters for IIter */
  virtual void SetParam(const configVec & conf) = 0;

  /* Initalize IIter */
  virtual void Initalize(void) = 0;

  /* Start IIter */
  virtual void StartIter(const int & iter) = 0;

  /* Advance iterator */
  virtual bool Next(void) = 0;

  /* Return latest entry */
  virtual const DataType & Entry(void) const = 0;

  virtual DataType & Entry(void) = 0;

  virtual ~IIter(void) {};

};

struct DataChunk {
  
public:
  DataChunk(void) {};
  /* Assignment constructor for deep copy */
  DataChunk & operator=(const DataChunk & src);
  /* Return number of datapoint int this chunk */
  unsigned int Size(void);
public:
  /* id for this batch */
  //unsigned int batchId;
  
  /* data ids */
  //std::vector< int > dataIds;
  
  /* labels */
  std::vector< int > Labels;
  
  /* Vector of data-points */
  mshadow::TensorContainer<mshadow::cpu, 4, float> Data;

};

DataChunk & DataChunk::operator=(const DataChunk & src) {
  /* Check for self assignment */
  if ( this == &src ) return *this;
  /* Free memory and resize */
  if ( Labels.size() != src.Labels.size() ) {
    Labels.resize(src.Labels.size());
  }
  if ( Data.shape_ != src.Data.shape_ ) {
    Data.Resize(src.Data.shape_ );
  }
  /* Copy data elements */
  Labels = src.Labels;
  mshadow::Copy(Data, src.Data);
  return *this;
}

unsigned int DataChunk::Size(void) {
  return Labels.size();
}

/* Create a IIter from config file*/
IIter< DataChunk > * CreateIter(std::string net, const configVec & conf, bool isTrain);

}// End iter-namespace



#endif /* ITER_BASE_H_ */
