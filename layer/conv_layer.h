/*
 * conv_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef CONV_LAYER_H_
#define CONV_LAYER_H_

#include "layer.h"

template<typename xpu>
class Conv_layer : public ILayer<xpu>{

public:

	Conv_layer(ILayer<xpu>* inputLayer, int ksize, int kstride, int nchannel):
			inputLayer_(inputLayer), ksize_(ksize), kstride_(kstride), nchannel_(nchannel), pskernelkey_(-1), psbiaskey_(-1), backPropError_(true) {	}

	void onBatchSizeChanged( int batch_size ){

		activations_.FreeSpace();
		activations_.data.shape_ = Shape4(batch_size,
										  nchannel_,
										  (inputLayer_->getpAct()->data.size(2) - ksize_)/ kstride_ + 1,
										  (inputLayer_->getpAct()->data.size(3) - ksize_)/ kstride_ + 1  );
		activations_.AllocSpace();

	}

	Node<xpu>* getpAct(void){
		return &activations_;
	}


	void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

		//init nodes without batchsize
		activations_.set_stream(stream);
		activations_.data.shape_ = Shape4(inputLayer_->getpAct()->data.size(0),
										  nchannel_,
										  (inputLayer_->getpAct()->data.size(2) - ksize_)/ kstride_ + 1,
										  (inputLayer_->getpAct()->data.size(3) - ksize_)/ kstride_ + 1  );


		//set weights
		kernel_.set_stream(stream);  gkernel_.set_stream(stream);
		bias_.set_stream(stream); gbias_.set_stream(stream);

		tmp_col_.set_stream(stream);
		tmp_dst_.set_stream(stream);

		kernel_.Resize(Shape2(nchannel_,inputLayer_->getpAct()->data.size(1)* ksize_* ksize_));  gkernel_.Resize(kernel_.shape_);
		rnd.SampleGaussian(&kernel_, 0, 0.1f);

		// setup bias
		bias_.Resize(Shape1(nchannel_)); gbias_.Resize(bias_.shape_);
		bias_ = 0.0f;
	}

	void feedforward(bool is_train){

	   // forward convolution, tmp_col_ and tmp_dst_ are helper structure
		index_t oheight  = (inputLayer_->getpAct()->data.size(2) - ksize_)/kstride_ + 1;
		index_t owidth   = (inputLayer_->getpAct()->data.size(3) - ksize_)/kstride_ + 1;
		index_t nbatch   = inputLayer_->getpAct()->data.size(0);

		// we directly unpack all local patches and do a dot product
		// this cost lots of memory, normally for large image, only unpack several image at a time
		tmp_col_.Resize(Shape2(inputLayer_->getpAct()->data.size(1)*ksize_*ksize_, nbatch*oheight*owidth));
		tmp_dst_.Resize(Shape2(nchannel_, nbatch*oheight*owidth));
		// unpack local patches , stride=1
		tmp_col_ = unpack_patch2col(inputLayer_->getpAct()->data, ksize_, ksize_, kstride_);

		tmp_dst_ = dot(kernel_, tmp_col_);

		// reshape, then swap axis, we chain equations together
		activations_.data = swapaxis<1,0>(reshape(tmp_dst_, Shape4(nchannel_, nbatch, oheight, owidth)));

		// add bias
		activations_.data += broadcast<1>(bias_, activations_.data.shape_);

	 }

	void backpropagate(void){


		index_t oheight  = (inputLayer_->getpAct()->data.size(2) - ksize_)/kstride_ + 1;
		index_t owidth   = (inputLayer_->getpAct()->data.size(3) - ksize_)/kstride_ + 1;
		index_t nbatch   = inputLayer_->getpAct()->data.size(0);

		// we directly unpack all local patches and do a dot product
		// this cost lots of memory, normally for large image, only unpack several image at a time
		tmp_col_.Resize(Shape2(inputLayer_->getpAct()->data.size(1) * ksize_ * ksize_,
							  nbatch * oheight * owidth));
		tmp_dst_.Resize(Shape2(nchannel_, nbatch * oheight * owidth));
		// unpack local patches
		tmp_col_ = unpack_patch2col(inputLayer_->getpAct()->data, ksize_, ksize_, kstride_);
		tmp_dst_ = reshape(swapaxis<1,0>(activations_.data), tmp_dst_.shape_);
		gkernel_ = dot(tmp_dst_, tmp_col_.T());

		if(backPropError_){
		// backpropgation: not necessary for first layer, but included anyway

		tmp_col_ = dot(kernel_.T(), tmp_dst_);
		inputLayer_->getpAct()->data = pack_col2patch(tmp_col_, inputLayer_->getpAct()->data.shape_, ksize_, ksize_, kstride_);
		// calc grad of bias
		}

		gbias_ = sumall_except_dim<1>(activations_.data);
	}


	void ApplyVisitor(IVisitor<xpu> *pvisitor) {
			pvisitor->Visit(false, &kernel_, &gkernel_);
			pvisitor->Visit(true, bias_, gbias_);
	 }

	 void setBackpropError(bool backpropError){
		 backPropError_ = backpropError;
	 }


	std::string getType(){
		return "conv";
	}
	int getParamSize(){
		return ( kernel_.size(0) * kernel_.size(1)  + bias_.size(0));
	}

	~Conv_layer(){
		activations_.FreeSpace();
	}

private:

  ILayer<xpu>* inputLayer_;
  Node<xpu> activations_;
  int ksize_, kstride_, nchannel_;

  bool backPropError_;

  // weight, gradient: kernel_ is actually convoltuion kernel, with shape=(num_channel,ksize*ksize)
  TensorContainer<xpu, 2, real_t> kernel_,  gkernel_;
  // temp helper structure
  TensorContainer<xpu, 2, real_t> tmp_col_, tmp_dst_;
  TensorContainer<xpu, 1, real_t> bias_, gbias_;

  int pskernelkey_;
  int psbiaskey_;

};


#endif /* CONV_LAYER_H_ */
