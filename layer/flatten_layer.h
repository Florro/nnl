/*
 * flatten_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef FLATTEN_LAYER_H_
#define FLATTEN_LAYER_H_

#include "layer.h"

template<typename xpu>
class flatten_layer : public ILayer<xpu>{

public:

	flatten_layer(ILayer<xpu>* inputLayer):
					inputLayer_(inputLayer), backPropError(true){	};

	void onBatchSizeChanged( int batch_size ){
		activations_.data.shape_ = Shape4(batch_size,
										 1,
										 1,
										 inputLayer_->getpAct()->data.size(1)*inputLayer_->getpAct()->data.size(2)*inputLayer_->getpAct()->data.size(3));
		activations_.AllocSpace();
	}

	void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){
		activations_.set_stream(stream);
		activations_.data.shape_ = Shape4(inputLayer_->getpAct()->data.size(0),
										  1,
										  1,
									      inputLayer_->getpAct()->data.size(1)*inputLayer_->getpAct()->data.size(2)*inputLayer_->getpAct()->data.size(3));

	}


	Node<xpu>* getpAct(void){
		return &activations_;
	}

	void feedforward(bool is_train){
		activations_.data = reshape(inputLayer_->getpAct()->data, activations_.data.shape_);
	 }

	void backpropagate(void){
		if(backPropError){
			inputLayer_->getpAct()->data = reshape(activations_.data, inputLayer_->getpAct()->data.shape_);
		}
	}
	~flatten_layer(){
		activations_.FreeSpace();
	}


	bool backPropError;
	std::string getType(){
		return "flat";
	}
	int getParamSize(){
		return (0);
	}



private:

	Node<xpu> activations_;
	ILayer<xpu>* inputLayer_;

};



#endif /* FLATTEN_LAYER_H_ */
