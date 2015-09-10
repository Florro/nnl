/*
 * conv_cudnn_layer.h
 *
 *  Created on: Aug 18, 2015
 *      Author: niklas
 */

#ifndef CONV_CUDNN_LAYER_H_
#define CONV_CUDNN_LAYER_H_

#include "layer.h"
#include <cudnn.h>

#define CUDNN_SAFE_CALL(call) do { cudnnStatus_t err = call; \
		if(err != CUDNN_STATUS_SUCCESS){ \
			printf("Cudnn error at %s:%d\nError: %s\n",__FILE__,__LINE__, cudnnGetErrorString(call)); \
			exit(EXIT_FAILURE);}} while(0)

template<typename xpu>
class Conv_cudnn_layer : public ILayer<xpu>{

public:

	Conv_cudnn_layer(ILayer<xpu>* inputLayer, int ksize, int kstride, int nchannel, int pad, real_t weightInit){}
	void onBatchSizeChanged( int batch_size ) {};
	std::string getType(){		return "conv";	}
	int getParamSize(){		return (0 );	}
	~Conv_cudnn_layer(){}
	void backpropagate(){};
	void feedforward(bool){};
	Node<xpu>* getpAct(void){
		return &activations_;
	};



private:

	Node<xpu> activations_;

};

template<>
class Conv_cudnn_layer<gpu> : public ILayer<gpu>{

public:

	Conv_cudnn_layer(ILayer<gpu>* inputLayer, int ksize, int kstride, int nchannel, int pad, real_t weightInit):
			inputLayer_(inputLayer), ksize_(ksize), kstride_(kstride), nchannel_(nchannel),
			pad_x_(pad), pad_y_(pad),
			weightInit_(weightInit),
			kernel_(false), bias_(false), gkernel_(false), gbias_(false),
			backPropError_(true) {}


	void onBatchSizeChanged( int batch_size ){

		activations_.FreeSpace();
		activations_.data.shape_ = Shape4(batch_size,
										  nchannel_,
										  (inputLayer_->getpAct()->data.size(2) + 2 * pad_y_ - ksize_)/ kstride_ + 1,
										  (inputLayer_->getpAct()->data.size(3) + 2 * pad_x_ - ksize_)/ kstride_ + 1  );
		activations_.AllocSpace();
	}



	Node<gpu>* getpAct(void){
		return &activations_;
	}


	void InitLayer(mshadow::Stream<gpu> *stream, Random<gpu, real_t> &rnd){

		this->InitCuDNN();

		//init nodes without batchsize
		activations_.set_stream(stream);
		activations_.data.shape_ = Shape4(inputLayer_->getpAct()->data.size(0),
										  nchannel_,
										  (inputLayer_->getpAct()->data.size(2) + 2 * pad_y_ - ksize_)/ kstride_ + 1,
										  (inputLayer_->getpAct()->data.size(3) + 2 * pad_x_ - ksize_)/ kstride_ + 1  );


		//set weights
		kernel_.set_stream(stream);  gkernel_.set_stream(stream);
		bias_.set_stream(stream); gbias_.set_stream(stream);

		kernel_.Resize(Shape4(nchannel_, inputLayer_->getpAct()->data.size(1), ksize_ , ksize_ ));  gkernel_.Resize(kernel_.shape_);


		// setup bias
		bias_.Resize(Shape1(nchannel_)); gbias_.Resize(bias_.shape_);
		bias_ = 0.0f;
		gbias_ = 0.0f;
		gkernel_ = 0.0f;

		//init weights
		if(weightInit_ == 0.0f){
			//Xavier initalization
			real_t a = sqrt(3.0f / (kernel_.size(2) + kernel_.size(1) + kernel_.size(3)));
			rnd.SampleUniform(&kernel_, -a, a);
		}else{
			//Gaussian
			rnd.SampleGaussian(&kernel_, 0, weightInit_);
		}

	}

	void feedforward(bool is_train){

		float alpha = 1.0f;
		float beta = 0.0f;

		temp_.set_stream(activations_.data.stream_);
		temp_.set_pad(false);

		if(!activations_.data.CheckContiguous()){
			printf("fucking fuck\n");
		}
		if(!inputLayer_->getpAct()->data.CheckContiguous()){
			printf("fucking fuck\n");
		}

		CUDNN_SAFE_CALL(cudnnSetStream(handle_, activations_.data.stream_->stream_));
		CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
												   dtype_,
		                                           nchannel_,
		                                           inputLayer_->getpAct()->data.size(1),
		                                           ksize_,
		                                           ksize_));
		CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc_,
														pad_y_, //padding y
														pad_x_, //padding x
														kstride_,
														kstride_,
														1, 1, //upscale
														CUDNN_CROSS_CORRELATION));

		mshadow::Tensor<gpu, 4, float> &in  = inputLayer_->getpAct()->data;
	    mshadow::Tensor<gpu, 4, float> &out = activations_.data;

	    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(in_desc_, CUDNN_TENSOR_NCHW, dtype_,
	                                               in.shape_[0], in.shape_[1],
	                                               in.shape_[2], in.shape_[3]));

	    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW, dtype_,
	                                               out.shape_[0], out.shape_[1],
	                                               out.shape_[2], out.shape_[3]));
	    CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, dtype_,
	    										   1,  bias_.shape_[0], 1, 1));


	    // cudnn v3
	    CUDNN_SAFE_CALL(cudnnGetConvolutionForwardAlgorithm(handle_,
														   in_desc_,
														   filter_desc_,
														   conv_desc_,
														   out_desc_,
														   CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
														   512<<20,
														   &algo_));

	    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
																  in_desc_,
																  out_desc_,
																  conv_desc_,
																  filter_desc_,
																  CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
																  512<<20,
																  &back_algo_w_));

	    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle_,
																  filter_desc_,
																  out_desc_,
																  conv_desc_,
																  in_desc_,
																  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
																  512<<20,
																  &back_algo_));

	    size_t back_size = 0;
	    size_t back_size_w = 0;
	    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
																	filter_desc_,
																	out_desc_,
																	conv_desc_,
																	in_desc_,
																	back_algo_,
																	&back_size));

	    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
																	  in_desc_,
																	  out_desc_,
																	  conv_desc_,
																	  filter_desc_,
																	  back_algo_w_,
																	  &back_size_w));
	    back_size = std::max(back_size, back_size_w);
	    CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle_, in_desc_,
																 filter_desc_, conv_desc_,
																 out_desc_, algo_,
																 &workspace_size_));
	    workspace_size_ = std::max(back_size, workspace_size_);
	    temp_.Resize(mshadow::Shape1(workspace_size_ / sizeof(float) + 1), 0.0f);


	    CUDNN_SAFE_CALL(cudnnConvolutionForward(handle_, &alpha,
	                                           in_desc_, inputLayer_->getpAct()->data.dptr_,
	                                           filter_desc_, kernel_.dptr_,
	                                           conv_desc_, algo_, temp_.dptr_, workspace_size_, &beta,
	                                           out_desc_, activations_.data.dptr_));


	    beta = 1.0f;
		CUDNN_SAFE_CALL(cudnnAddTensor(handle_, CUDNN_ADD_SAME_C, &alpha,
												bias_desc_, bias_.dptr_, &beta,
												out_desc_, activations_.data.dptr_));


		/*
		TensorContainer<cpu, 4, real_t> data;
		data.Resize(activations_.data.shape_);
		Copy(data, activations_.data, activations_.data.stream_);
		for(int i = 128; i < 140; i++){
			for(int j = 128; j < 140; j++){
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




		/*
		TensorContainer<cpu, 4, real_t> data;
		data.Resize(inputLayer_->getpAct()->data.shape_);
		Copy(data, inputLayer_->getpAct()->data, inputLayer_->getpAct()->data.stream_);
		for(int i = 0; i < data.size(2); i++){
			for(int j = 0; j < data.size(3); j++){
				std::cout <<  data[0][2][i][j] << " ";

			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		std::cout << "cout finnished" << std::endl;
		*/




	 }

	void backpropagate(void){


 		float alpha = 1.0f;
		float beta = 0.0f;
		//CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1

		CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter_v3(handle_, &alpha,
												  in_desc_, inputLayer_->getpAct()->data.dptr_,
												  out_desc_, activations_.data.dptr_,
												  conv_desc_, back_algo_w_, temp_.dptr_, workspace_size_, &beta,
												  filter_desc_, gkernel_.dptr_));

		//CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter( handle_, &alpha, in_desc_, inputLayer_->getpAct()->data.dptr_,
		//												out_desc_, activations_.data.dptr_, conv_desc_,&beta,filter_desc_, gkernel_.dptr_));



		if(backPropError_){//CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
			CUDNN_SAFE_CALL(cudnnConvolutionBackwardData_v3(handle_, &alpha,
													filter_desc_, kernel_.dptr_,
													out_desc_, activations_.data.dptr_,
													conv_desc_, back_algo_, temp_.dptr_, workspace_size_, &beta,
													in_desc_, inputLayer_->getpAct()->data.dptr_));
		}

		//CUDNN_SAFE_CALL(cudnnConvolutionBackwardData( handle_, &alpha, filter_desc_, kernel_.dptr_,
		//		out_desc_, activations_.data.dptr_, conv_desc_,&beta, in_desc_, inputLayer_->getpAct()->data.dptr_));



		CUDNN_SAFE_CALL(cudnnConvolutionBackwardBias(handle_, &alpha,
													  out_desc_, activations_.data.dptr_,
													  &beta,
													  bias_desc_, gbias_.dptr_));

	}


	void ApplyVisitor(IVisitor<gpu> *pvisitor) {
			pvisitor->Visit(false, kernel_, gkernel_);
			pvisitor->Visit(true, bias_, gbias_);
	 }

	 ~Conv_cudnn_layer() {

		 activations_.FreeSpace();

		 CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(in_desc_));
		 CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(out_desc_));
		 CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
		 CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
		 CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
		 CUDNN_SAFE_CALL(cudnnDestroy(handle_));
	  };

	 void setBackpropError(bool backpropError){
		 backPropError_ = backpropError;
	 }

	 std::string getType(){
	 	return "conv";
	 }
	 int getParamSize(){
	 	return ( kernel_.size(0) * kernel_.size(1) * kernel_.size(2) * kernel_.size(3)  + bias_.size(0));
	 }



private:

  ILayer<gpu>* inputLayer_;
  Node<gpu> activations_;
  int ksize_, kstride_, nchannel_, pad_x_, pad_y_;
  real_t weightInit_;

  // weight, gradient: kernel_ is actually convoltuion kernel
  TensorContainer<gpu, 4, real_t> kernel_,  gkernel_;
  TensorContainer<gpu, 1, real_t> bias_, gbias_;

  bool backPropError_;

  inline void InitCuDNN() {
      dtype_ = CUDNN_DATA_FLOAT;
      algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      CUDNN_SAFE_CALL(cudnnCreate(&handle_));
      CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&in_desc_));
      CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&out_desc_));
      CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
      CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
      CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    }

  /*! \brief cuDNN init status */
  bool init_cudnn_;
  /*! \brief cuDNN handle */
  cudnnHandle_t handle_;
  /*! \brief cuDNN data type */
  cudnnDataType_t dtype_;
  /*! \brief cuDNN input tensor descriptor */
  cudnnTensorDescriptor_t in_desc_;
  /*! \brief cuDNN output tensor descriptor */
  cudnnTensorDescriptor_t out_desc_;
  /*! \brief cuDNN bias tensor descriptor */
  cudnnTensorDescriptor_t bias_desc_;
  /*! \brief cuDNN filter descriptor */
  cudnnFilterDescriptor_t filter_desc_;
  /*! \brief cuDNN conv descriptor */
  cudnnConvolutionDescriptor_t conv_desc_;
  /*! \brief cuDNN conv algorithm */
  cudnnConvolutionFwdAlgo_t algo_;
  /*! \brief cuDNN back algo for data */
  cudnnConvolutionBwdDataAlgo_t back_algo_;
  /*! \brief cuDNN back algo for filter */
  cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  /*! \brief cuDNN workspace size */
  size_t workspace_size_;
  /*! \brief cuDNN workspace */
  mshadow::TensorContainer<gpu, 1> temp_;

};


#endif /* CONV_CUDNN_LAYER_H_ */
