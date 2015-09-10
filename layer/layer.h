/*
 * layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef LAYER_H_

#include "mshadow/tensor.h"
#include "mshadow-ps/mshadow_ps.h"
#include "string.h"
#include "../utility/util.h"


using namespace mshadow;
using namespace mshadow::expr;

template<typename xpu>
struct Node {
  /*!
   * \brief content of the node
   *  layout:
   *     images (batch_size, nchannel, height, width)
   *     matrix (batch_size, 1, 1, length-of-vector)
   */
  mshadow::Tensor<xpu, 4> data;
  /*! \brief whether the underlying data must be contiguous */
  bool must_contiguous;
  bool inited;

  // constructor
  Node(void) : must_contiguous(true) {
    data.shape_ = mshadow::Shape4(0,0,0,0);
    inited = false;
  }
  /*! \brief matrix view of the node */
  inline mshadow::Tensor<xpu, 2> mat(void) {
    return data.FlatTo2D();
  }
  /*! \brief check whether it holds a matrix data */
  inline bool is_mat(void) const {
    return data.size(1) == 1 && data.size(2) == 1;
  }

  inline void set_stream(mshadow::Stream<xpu> *stream){
	  data.set_stream(stream);
  }

  /*! \brief helper rountine to free space */
  inline void FreeSpace(void) {
    if (inited){
      mshadow::FreeSpace(&data);
    }
  }
  /*! \brief helper rountine to allocate space */
  inline void AllocSpace(void) {
    if (must_contiguous) {
      mshadow::AllocSpace(&data, false);
      if(!data.CheckContiguous()){
    	  printf("Error: Data not continuous!\n");
      }
    } else {
      mshadow::AllocSpace(&data);

    }
    inited = true;
  }
}; // struct Node



template<typename xpu>
class IVisitor{
public:
	/*
	 * \brief visit content of the layer, this is called by Layer
	 *    when ApplyVisitor is called
	 *
	 *    use to interact with weights.
	 *
	*/

	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu, 1> weight,
					   mshadow::Tensor<xpu, 1> grad) = 0;
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu, 2> weight,
					   mshadow::Tensor<xpu, 2> grad) = 0;
	virtual void Visit(bool is_bias,
			 	 	   mshadow::Tensor<xpu, 3> weight,
					   mshadow::Tensor<xpu, 3> grad) = 0;
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu, 4> weight,
					   mshadow::Tensor<xpu, 4> grad) = 0;

	virtual void setEpoch(int epoch) {}
	virtual bool is_weight_Visitor() { return true; }
	virtual ~IVisitor(){};

};

enum visitormode{
	INITKEY = 0,
	PULLREQ = 1,
	PULLWAIT = 2
};



template<typename xpu>
class ILayer{

public:

	virtual void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd) {};
	virtual void feedforward(bool is_train) = 0;
	virtual void backpropagate(void) = 0;
	virtual void ApplyVisitor( IVisitor<xpu> *pvisitor ) {}
	virtual void setBackpropError(bool backpropError) {};
	virtual Node<xpu>* getpAct(void) = 0;
	virtual std::string getType() = 0;
	virtual int getParamSize() = 0;

	virtual void onBatchSizeChanged(int batch_size) {};
	virtual ~ILayer(){};
};



#define LAYER_H_




#endif /* LAYER_H_ */
