/*
 * standard_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef STANDARD_LAYER_H_
#define STANDARD_LAYER_H_

#include "layer.h"

template<typename xpu>
class Standard_layer : public ILayer<xpu>{

public:

	Standard_layer(ILayer<xpu>* inputLayer, int numHidden):
			inputLayer_(inputLayer), numHidden_(numHidden),
			backPropError_(true){}

	void onBatchSizeChanged( int batch_size ){
		activations_.data.shape_ = Shape4(batch_size,
											 1,
											 1,
											 numHidden_);
		activations_.AllocSpace();
	}

	Node<xpu>* getpAct(void){
		return &activations_;
	}

	void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

		utility::Check(inputLayer_->getpAct()->is_mat(), "Standardlayer InputNode is not a Matrix");

		activations_.set_stream(stream);
		activations_.data.shape_ = Shape4(inputLayer_->getpAct()->data.size(0),
												 1,
												 1,
												 numHidden_);

		wmat_.set_stream(stream); gwmat_.set_stream(stream);
		bias_.set_stream(stream); gbias_.set_stream(stream);

		wmat_.Resize(Shape2(numHidden_, inputLayer_->getpAct()->data.size(3)));   gwmat_.Resize(wmat_.shape_);
		bias_.Resize(Shape1(numHidden_));                			    		  gbias_.Resize(bias_.shape_);

		//initialize
		bias_ = 0.0f;
		gwmat_ = 0.0f;
		gbias_ = 0.0f;


		//Gaussian
		//rnd.SampleGaussian(&wmat_, 0, weightsInit_);

		//Xavier initialization
		real_t a = sqrt(3.0f / (wmat_.size(1) + wmat_.size(0)));
	    rnd.SampleUniform(&wmat_, -a, a);



	}

	void feedforward(bool is_train){

		mshadow::Tensor<xpu, 2> m_in = inputLayer_->getpAct()->mat();
		mshadow::Tensor<xpu, 2> m_out = activations_.mat();

		index_t batch_size = m_in.size(0);

		m_out = dot(m_in, wmat_.T());
		m_out += repmat(bias_, batch_size);

	}
	void backpropagate(void){

		mshadow::Tensor<xpu, 2> m_in = inputLayer_->getpAct()->mat();
		mshadow::Tensor<xpu, 2> m_out = activations_.mat();

		// accumulate gradient
     	gwmat_ = dot(m_out.T(), m_in);
		gbias_ = sum_rows(m_out);
		// backprop
		if(backPropError_){
			m_in = dot(m_out, wmat_);
		}
	};

	void ApplyVisitor(IVisitor<xpu> *pvisitor) {
		pvisitor->Visit(false, wmat_, gwmat_);
		pvisitor->Visit(true, bias_, gbias_);
	 }


	std::string getType(){
		return "full";
	}
	int getParamSize(){
		return ( wmat_.size(0) * wmat_.size(1) + bias_.size(0) );
	}
	void setBackpropError(bool backpropError){
	 backPropError_ = backpropError;
	}

	~Standard_layer(){
		activations_.FreeSpace();
	}

private:

	Node<xpu> activations_;

	int numHidden_;

	bool backPropError_;

	ILayer<xpu>* inputLayer_;

	mshadow::TensorContainer<xpu,2> wmat_, gwmat_;
	mshadow::TensorContainer<xpu,1> bias_, gbias_;

};


#endif /* STANDARD_LAYER_H_ */
