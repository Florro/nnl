/*
 * pool_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef POOL_LAYER_H_
#define POOL_LAYER_H_

#include "layer.h"

#define CUDNN_SAFE_CALL(call) do { cudnnStatus_t err = call; \
		if(err != CUDNN_STATUS_SUCCESS){ \
			printf("Cudnn error at %s:%d\nError: %s\n",__FILE__,__LINE__, cudnnGetErrorString(call)); \
			exit(EXIT_FAILURE);}} while(0)

template<typename xpu, typename pooltype>
class pool_layer : public ILayer<xpu>{

public:



	pool_layer(ILayer<xpu>* inputLayer, int psize, int pstride):
			psize_(psize), inputLayer_(inputLayer), pstride_(pstride), backPropError(true){	};

	void onBatchSizeChanged( int batch_size ){

		activations_.data.shape_ = Shape4(batch_size,
										  inputLayer_->getpAct()->data.size(1),
										  std::min(inputLayer_->getpAct()->data.size(2) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(2) - 1) / pstride_ + 1,
										  std::min(inputLayer_->getpAct()->data.size(3) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(3) - 1) / pstride_ + 1);

		activations_.AllocSpace();

	}

	void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){
		this->stream_ = stream;
		activations_.set_stream(stream);

		activations_.data.shape_ = Shape4(inputLayer_->getpAct()->data.size(0),
										  inputLayer_->getpAct()->data.size(1),
										  std::min(inputLayer_->getpAct()->data.size(2) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(2) - 1) / pstride_ + 1,
										  std::min(inputLayer_->getpAct()->data.size(3) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(3) - 1) / pstride_ + 1);

	}

	Node<xpu>* getpAct(void){
		return &activations_;
	}

	void feedforward(bool is_train){
		if(pooltmp_.shape_ != activations_.data.shape_){
			pooltmp_.Resize(activations_.data.shape_); //workaround for connectstate
		}

		// max pooling
		activations_.data = pool<pooltype>(inputLayer_->getpAct()->data, activations_.data[0][0].shape_, psize_, psize_, pstride_);
		Copy(pooltmp_, activations_.data, stream_);


		/*
		TensorContainer<cpu, 4, real_t> data;
		data.Resize(activations_.data.shape_);
		Copy(data, activations_.data, activations_.data.stream_);
		for(int i = 0; i < data.size(2); i++){
			for(int j = 0; j < data.size(3); j++){
				std::cout <<  data[0][0][i][j] << " ";

			}
			std::cout << std::endl;
		}
		std::cout << "tens1" << std::endl;
		int a;
		std::cin >> a;
		std::cin.clear();
		std::cin.ignore(INT_MAX,'\n');
		*/

	 }

	void backpropagate(void){
		if(backPropError){
			inputLayer_->getpAct()->data = unpool<red::maximum>(inputLayer_->getpAct()->data, pooltmp_, activations_.data, psize_, psize_, pstride_);
		}
	}

	bool backPropError;
	std::string getType(){
		return "pool";
	}
	int getParamSize(){
		return ( 0);
	}

	~pool_layer(){
		activations_.FreeSpace();
	}

private:

	ILayer<xpu>* inputLayer_;
	Node<xpu> activations_;
	int psize_, pstride_;
	mshadow::Stream<xpu> *stream_;
	mshadow::TensorContainer<xpu, 4> pooltmp_;

};


#endif /* POOL_LAYER_H_ */
