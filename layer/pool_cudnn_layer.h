/*
 * pool_cudnn_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef POOL_CUDNN_LAYER_H_
#define POOL_CUDNN_LAYER_H_

#include "layer.h"
#include <cudnn.h>

enum poolmode{
	MAX = 0,
	AVG =  1
};


template<typename xpu, int mode>
class pool_cudnn_layer : public ILayer<xpu>{

public:

	pool_cudnn_layer(ILayer<xpu>* inputLayer, int psize, int pstride){}
	void onBatchSizeChanged( int batch_size ) {};
	Node<xpu>* getpAct(void){
		return &activations_;
	};
	std::string getType(){		return "pool";	}
	int getParamSize(){		return (0 );	}
	void feedforward(bool){}
	void backpropagate(){}
	~pool_cudnn_layer(){};

private:

	Node<xpu> activations_;

};


template<int mode>
class pool_cudnn_layer<gpu, mode> : public ILayer<gpu>{

public:

	bool backPropError;

	pool_cudnn_layer(ILayer<gpu>* inputLayer, int psize, int pstride):
			psize_(psize), inputLayer_(inputLayer), pstride_(pstride), backPropError(true){};

	void onBatchSizeChanged( int batch_size ){
		activations_.FreeSpace();
		temp_.FreeSpace();
		temp2_.FreeSpace();

		activations_.data.shape_ = Shape4(batch_size,
										  inputLayer_->getpAct()->data.size(1),
										  std::min(inputLayer_->getpAct()->data.size(2) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(2) - 1) / pstride_ + 1,
										  std::min(inputLayer_->getpAct()->data.size(3) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(3) - 1) / pstride_ + 1);


		temp_.data.shape_ = activations_.data.shape_;
		temp2_.data.shape_ = inputLayer_->getpAct()->data.shape_;


		activations_.AllocSpace();
		temp_.AllocSpace();
		temp2_.AllocSpace();

	}

	void InitLayer(mshadow::Stream<gpu> *stream, Random<gpu, real_t> &rnd){
		this->InitCuDNN();
		this->stream_ = stream;

		activations_.set_stream(stream);
		activations_.data.shape_ = Shape4(inputLayer_->getpAct()->data.size(0),
										  inputLayer_->getpAct()->data.size(1),
										  std::min(inputLayer_->getpAct()->data.size(2) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(2) - 1) / pstride_ + 1,
										  std::min(inputLayer_->getpAct()->data.size(3) - psize_ + pstride_ - 1, inputLayer_->getpAct()->data.size(3) - 1) / pstride_ + 1);

		//pooltmp_.set_stream(stream_);
		//pooltmp_2.set_stream(stream_);

		temp_.set_stream(stream_);
		temp_.data.shape_ = activations_.data.shape_;

		temp2_.set_stream(stream_);
		temp2_.data.shape_ = inputLayer_->getpAct()->data.shape_;


	}

	Node<gpu>* getpAct(void){
		return &activations_;
	}

	void feedforward(bool is_train){

		CUDNN_SAFE_CALL(cudnnSetStream(handle_, activations_.data.stream_->stream_));
		mshadow::Tensor<gpu, 4, float> &in =  inputLayer_->getpAct()->data;
		mshadow::Tensor<gpu, 4, float> &out = activations_.data;
		CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(in_desc_, CUDNN_TENSOR_NCHW, dtype_,
											  in.shape_[0], in.shape_[1],
											  in.shape_[2], in.shape_[3]));
		CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW, dtype_,
											  out.shape_[0], out.shape_[1],
											  out.shape_[2], out.shape_[3]));


		float alpha = 1.0f;
	    float beta = 0.0f;

	    utility::Check(inputLayer_->getpAct()->data.CheckContiguous(), "pooling input data not contiguous");
	    utility::Check(activations_.data.CheckContiguous(), "pooling data not contiguous");
	    utility::Check(temp_.data.CheckContiguous(), "pooling temp1 data not contiguous");

	    CUDNN_SAFE_CALL(cudnnPoolingForward( handle_, pooling_desc_, &alpha,
									 	 	 in_desc_, inputLayer_->getpAct()->data.dptr_, &beta,
									 	 	 out_desc_, temp_.data.dptr_));

		// copy into temps
		Copy(activations_.data, temp_.data, stream_);


	 }

	void backpropagate(void){

		float alpha = 1.0f;
		float beta = 0.0f;

		utility::Check(temp2_.data.CheckContiguous(), "pooling temp2 data not contiguous");

		if(backPropError){
		CUDNN_SAFE_CALL(cudnnPoolingBackward(	handle_, pooling_desc_, &alpha,
												out_desc_, temp_.data.dptr_,
												out_desc_, activations_.data.dptr_,
												in_desc_, inputLayer_->getpAct()->data.dptr_, &beta,
												in_desc_, temp2_.data.dptr_	));
		// copy into temps
		Copy(inputLayer_->getpAct()->data, temp2_.data, stream_);
		}
	}

	~pool_cudnn_layer(void) {

		activations_.FreeSpace();
		temp_.FreeSpace();
		temp2_.FreeSpace();

		CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(in_desc_));
		CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(out_desc_));
		CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
		CUDNN_SAFE_CALL(cudnnDestroy(handle_));
	}

	std::string getType(){
		return "pool";
	}
	int getParamSize(){
		return ( 0 );
	}



private:

	ILayer<gpu>* inputLayer_;
	Node<gpu> activations_;
	int psize_;
	int pstride_;
	mshadow::Stream<gpu> *stream_;

	Node<gpu> temp_;
	Node<gpu> temp2_;

	//mshadow::TensorContainer<gpu, 4> pooltmp_;
	//mshadow::TensorContainer<gpu, 4> pooltmp_2;

protected:

    inline void InitCuDNN() {

      dtype_ = CUDNN_DATA_FLOAT;
      if(mode == MAX){
    	  mode_ = CUDNN_POOLING_MAX;
      }else if (mode == AVG){
    	  mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      }
      else{
    	  utility::Error("Error invalid pooling mode!");
      }
      CUDNN_SAFE_CALL(cudnnCreate(&handle_));
      CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&in_desc_));
      CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&out_desc_));
      CUDNN_SAFE_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
      CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(pooling_desc_, mode_,
                                             psize_,
                                             psize_,
                                             0, 0, //padding x,y
                                             pstride_, pstride_  //stride x,y
                                             ));

    }

    /*! \brief cuDNN data type */
    cudnnDataType_t dtype_;
    /*! \brief cudnn handle */
    cudnnHandle_t handle_;
    /*! \brief cudnn pooling mode */
    cudnnPoolingMode_t mode_;
    /*! \brief input descriptor */
    cudnnTensorDescriptor_t in_desc_;
    /*! \brief output descriptor */
    cudnnTensorDescriptor_t out_desc_;
    /*! \brief pooling descriptor */
    cudnnPoolingDescriptor_t pooling_desc_;


};


#endif /* POOL_CUDNN_LAYER_H_ */
