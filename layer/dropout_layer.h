/*
 * dropout_layer.h
 *
 *  Created on: Aug 20, 2015
 *      Author: niklas
 */

#ifndef DROPOUT_LAYER_H_
#define DROPOUT_LAYER_H_

#include "layer.h"
#include "../utility/mshadow_op.h"

//Dropout layer is a self-loop layer
template<typename xpu>
class Dropout_layer : public ILayer<xpu>{

public:

	Dropout_layer(ILayer<xpu>* inputLayer, float dropout):
			inputLayer_(inputLayer), dropout_(dropout), rnd_(0){}

	void onBatchSizeChanged( int batch_size ){

		mask_.Resize(inputLayer_->getpAct()->data.shape_);

	}


	void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

		mask_.set_stream(stream);

	}

	void feedforward(bool is_train){

		using namespace mshadow::expr;
		const real_t pkeep = 1.0f - dropout_;
		if(is_train){
			mask_ = F<threshold>(rnd_.uniform(mask_.shape_), pkeep)  * (1.0f/pkeep);
			inputLayer_->getpAct()->data = inputLayer_->getpAct()->data * mask_;
		}



	}
	void backpropagate(void){

		using namespace mshadow::expr;
		inputLayer_->getpAct()->data = inputLayer_->getpAct()->data * mask_;

	};

	~Dropout_layer(){}

	Node<xpu>* getpAct(void){
		return inputLayer_->getpAct();
	}
	std::string getType(){
		return "drop";
	}
	int getParamSize(){
		return ( 0 );
	}



private:

	Node<xpu> activations_;
	mshadow::TensorContainer<xpu,4> mask_;
	Random<xpu, real_t> rnd_;
	float dropout_;

	ILayer<xpu>* inputLayer_;

};


#endif /* DROPOUT_LAYER_H_ */
