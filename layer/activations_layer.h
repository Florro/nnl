/*
 * activations_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef ACTIVATIONS_LAYER_H_
#define ACTIVATIONS_LAYER_H_

#include "layer.h"


template<typename xpu, typename fwdOp, typename BackOp>
class activation_layer : public ILayer<xpu>{

public:

	activation_layer(ILayer<xpu>* inputLayer):
						inputLayer_(inputLayer) {}

	void onBatchSizeChanged( int batch_size ){
		activations_.FreeSpace();
		activations_.data.shape_ = Shape4(batch_size,
										  inputLayer_->getpAct()->data.size(1),
										  inputLayer_->getpAct()->data.size(2),
										  inputLayer_->getpAct()->data.size(3));
		activations_.AllocSpace();
	}

	Node<xpu>* getpAct(void){
		return &activations_;
	}

	void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){
		stream_ = stream;
		activations_.set_stream(stream_);
		activations_.data.shape_ = Shape4(inputLayer_->getpAct()->data.size(0),
										  inputLayer_->getpAct()->data.size(1),
										  inputLayer_->getpAct()->data.size(2),
										  inputLayer_->getpAct()->data.size(3));

	}

	void feedforward(bool is_train){
		// relu layer
		inputLayer_->getpAct()->data = F<fwdOp>(inputLayer_->getpAct()->data);
		Copy(activations_.data, inputLayer_->getpAct()->data, stream_);
	 }

	void backpropagate(void){
		inputLayer_->getpAct()->data = F<BackOp>(inputLayer_->getpAct()->data) * activations_.data;
	}

	std::string getType(){
		return "act ";
	}
	int getParamSize(){
		return 0;
	}

	~activation_layer(){
		activations_.FreeSpace();
	}


private:

	ILayer<xpu>* inputLayer_;
	Node<xpu> activations_;
	mshadow::Stream<xpu> *stream_;

};


#endif /* ACTIVATIONS_LAYER_H_ */
