/*
 * data_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef DATA_LAYER_H_
#define DATA_LAYER_H_

#include "layer.h"

template<typename xpu>
class Data_layer : public ILayer<xpu>{

public:

	Data_layer(int c, int x, int y):
					c_(c), x_(x), y_(y) {};

	~Data_layer(){
		activations_.FreeSpace();
	};

	Node<xpu>* getpAct(void){
		return &activations_;
	}

	void onBatchSizeChanged( int batch_size ){
		activations_.FreeSpace();
		activations_.data.shape_ = Shape4(batch_size,c_,y_,x_);
		activations_.AllocSpace();
	}

	void feedforward(bool is_train){
		activations_.data = activations_.data / 256.0f; //toDo HARDCODE
	}


	void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){
		activations_.set_stream(stream);
		activations_.data.shape_ = Shape4(1,c_,x_,y_);
	}


	std::string getType(){
		return "data";
	}
	int getParamSize(){
		return ( 0 );
	}



private:

	int c_, x_, y_;

	Node<xpu> activations_;

};



#endif /* DATA_LAYER_H_ */
