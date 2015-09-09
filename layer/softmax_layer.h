/*
 * Softmax_layer.h
 *
 *  Created on: Aug 21, 2015
 *      Author: niklas
 */

#ifndef SOFTMAX_LAYER_H_
#define SOFTMAX_LAYER_H_

template<typename xpu>
class Loss_layer{
public:
	virtual void feedforward() = 0;
	virtual void backpropagate(int* labels) = 0;
	virtual std::string getType() = 0;
	virtual ~Loss_layer(){};
};




//loss layer are self-loop layer
template<typename xpu>
class Softmax_layer : public Loss_layer<xpu>{

public:

	Softmax_layer(ILayer<xpu>* inputLayer):
					inputLayer_(inputLayer){}

	std::string getType(){
		return "Softmax";
	}

	void feedforward(){
		Softmax(inputLayer_->getpAct()->mat(), inputLayer_->getpAct()->mat());
	}

	void backpropagate(int* labels){

		cpu_pred_.Resize(inputLayer_->getpAct()->mat().shape_);
		int batchSize_ = cpu_pred_.size(0);
		mshadow::Copy(cpu_pred_, inputLayer_->getpAct()->mat(), inputLayer_->getpAct()->data.stream_);
		//calculate gradient from label
		for (int k = 0; k < batchSize_; ++k) {
			cpu_pred_[k][ labels[k] ] -= 1.0f;
		}
		//scale gradient
		cpu_pred_ *= 1.0 / (batchSize_);
		mshadow::Copy(inputLayer_->getpAct()->mat(),cpu_pred_,  inputLayer_->getpAct()->data.stream_);

	}

	~Softmax_layer(){}

private:

	mshadow::TensorContainer<cpu, 2> cpu_pred_;
	ILayer<xpu>* inputLayer_;

};

#endif /* SOFTMAX_LAYER_H_ */
