/*
 * batchnorm_layer.h
 *
 *  Created on: Sep 9, 2015
 *      Author: niklas
 */

#ifndef BATCHNORM_LAYER_H_
#define BATCHNORM_LAYER_H_

#include "layer.h"


template<typename xpu, bool moving_avg>
class batchnorm_layer : public ILayer<xpu> {
 public:
  batchnorm_layer(ILayer<xpu>* inputLayer, real_t init_slope, real_t init_bias, real_t eps, real_t bn_momentum) :
	  inputLayer_(inputLayer), init_slope_(init_slope), init_bias_(init_bias), eps_(eps), bn_momentum_(bn_momentum){

  }

  virtual void ApplyVisitor(IVisitor<xpu> *pvisitor) {
    pvisitor->Visit(false, slope_, gslope_);
    pvisitor->Visit(true, bias_, gbias_);
  }

  void onBatchSizeChanged( int batch_size ){
	  	activations_.FreeSpace();
	  	activations_.data.shape_ = inputLayer_->getpAct()->data.shape_;
		activations_.AllocSpace();

	    temp_.Resize(inputLayer_->getpAct()->data.shape_);
	    in_shape_ = inputLayer_->getpAct()->data.shape_;
  }

  Node<xpu>* getpAct(void){
		return &activations_;
  }

  std::string getType(){
  		return "norm";
  }

  int getParamSize(){
  		return ( slope_.size(0) + bias_.size(0) );
  }

  ~batchnorm_layer(){
	  activations_.FreeSpace();
  }

  virtual void InitLayer(mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

	activations_.set_stream(stream);
	slope_.set_stream(stream);
	gslope_.set_stream(stream);
	exp_.set_stream(stream);
	gexp_.set_stream(stream);
	var_.set_stream(stream);
	gvar_.set_stream(stream);
	wtf_.set_stream(stream);
	bias_.set_stream(stream);
	gbias_.set_stream(stream);
	temp_.set_stream(stream);

	if (moving_avg) {
	  running_exp_.set_stream(stream);
	  running_var_.set_stream(stream);
	}

    in_shape_ = inputLayer_->getpAct()->data.shape_;

    if (in_shape_[1] == 1){
      // This is a fc layer
      channel_ = inputLayer_->getpAct()->data.size(3);
    } else {
      // This is a conv layer
      channel_ = inputLayer_->getpAct()->data.size(1);
    }
    activations_.data.shape_ = inputLayer_->getpAct()->data.shape_;

    slope_.Resize(mshadow::Shape1(channel_));
    gslope_.Resize(mshadow::Shape1(channel_));
    exp_.Resize(mshadow::Shape1(channel_));
    var_.Resize(mshadow::Shape1(channel_));
    gexp_.Resize(slope_.shape_);
    gvar_.Resize(slope_.shape_);
    wtf_.Resize(slope_.shape_);
    bias_.Resize(slope_.shape_);
    gbias_.Resize(slope_.shape_);
    if (moving_avg) {
 	 running_exp_.Resize(slope_.shape_);
 	 running_var_.Resize(slope_.shape_);
	 running_exp_ = 0.0f;
	 running_var_ = 0.0f;
    }
    gslope_ = 0.0f;
    gbias_ = 0.0f;
    gexp_ = 0.0f;
    gvar_ = 0.0f;
    slope_ = init_slope_;
    bias_ = init_bias_;

  }


  virtual void feedforward(bool is_train) {

    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = inputLayer_->getpAct()->data;
    mshadow::Tensor<xpu, 4> &out = activations_.data;

    float scale = 1.0f / in.shape_.Size() * channel_;

    if (is_train) {
      mshadow::Copy(temp_, in, temp_.stream_);
      if (in.size(1) != 1) {
        exp_ = scale * sumall_except_dim<1>(in);
        var_ = scale * sumall_except_dim<1>(F<square>(in - broadcast<1>(exp_, in.shape_)));
        in = (in - broadcast<1>(exp_, in.shape_)) / F<square_root>(broadcast<1>(var_ + eps_, in_shape_));
        out = in * broadcast<1>(slope_, in.shape_) + broadcast<1>(bias_, in.shape_);

      } else {
        exp_ = scale * sumall_except_dim<3>(in);
        var_ = scale * sumall_except_dim<3>(F<square>(in - broadcast<3>(exp_, in.shape_)));
        in = (in - broadcast<3>(exp_, in.shape_)) /  F<square_root>(broadcast<3>(var_ + eps_, in_shape_));
        out = in * broadcast<3>(slope_, in.shape_) + broadcast<3>(bias_, in.shape_);
      }
      if (moving_avg) {
        running_exp_ = running_exp_ * bn_momentum_ + exp_ * (1 - bn_momentum_);
        running_var_ = running_var_ * bn_momentum_ + var_ * (1 - bn_momentum_);
      }
    } else {
      if (in.size(1) != 1) {
        if (moving_avg) {
          out = broadcast<1>(slope_ / F<square_root>(running_var_ + eps_), in.shape_) *
            in + broadcast<1>(bias_ - (slope_ * running_exp_) /
                            F<square_root>(running_var_ + eps_), in.shape_);

        } else {
          exp_ = scale * sumall_except_dim<1>(in);
          var_ = scale * sumall_except_dim<1>(F<square>(in - broadcast<1>(exp_, in.shape_)));
          out = broadcast<1>(slope_ / F<square_root>(var_ + eps_), in.shape_) *
            in + broadcast<1>(bias_ - (slope_ * exp_) /
                            F<square_root>(var_ + eps_), in.shape_);
        }
      } else {
        if (moving_avg) {
          out = broadcast<3>(slope_ / F<square_root>(running_var_  + eps_), in.shape_) *
            in + broadcast<3>(bias_ - (slope_ * running_exp_) /
                            F<square_root>(running_var_ + eps_), in.shape_);
        } else {
          exp_ = scale * sumall_except_dim<3>(in);
          var_ = scale * sumall_except_dim<3>(F<square>(in - broadcast<3>(exp_, in.shape_)));
          out = broadcast<3>(slope_ / F<square_root>(var_  + eps_), in.shape_) *
            in + broadcast<3>(bias_ - (slope_ * exp_) /
                            F<square_root>(var_ + eps_), in.shape_);
        }
      }
    }
  }
  virtual void backpropagate() {

    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = inputLayer_->getpAct()->data;
    mshadow::Tensor<xpu, 4> &out = activations_.data;
    float scale = 1.0f / in.shape_.Size() * channel_;

    if (in.size(1) != 1){
      gvar_ = sumall_except_dim<1>((out * broadcast<1>(slope_, in.shape_)) *
                        (temp_ - broadcast<1>(exp_, in.shape_)) *
                        -0.5f * F<power>(broadcast<1>(var_ + eps_, in.shape_), -1.5f));
      gexp_ = sumall_except_dim<1>(out * broadcast<1>(slope_, in.shape_));
      gexp_ *= -1.0f / F<square_root>(var_ + eps_);
      wtf_ = scale * sumall_except_dim<1>(-2.0f * (temp_ - broadcast<1>(exp_, in.shape_)));
      wtf_ *= gvar_;
      gexp_ += wtf_;
      gslope_ += sumall_except_dim<1>(out * in);
      gbias_ += sumall_except_dim<1>(out);
      in = (out * broadcast<1>(slope_, in.shape_)) *
           broadcast<1>(1.0f / F<square_root>(var_ + eps_), in.shape_) +
           broadcast<1>(gvar_, in.shape_) * scale * 2.0f * (temp_ - broadcast<1>(exp_, in.shape_)) +
           broadcast<1>(gexp_, in.shape_) * scale;
    } else {
      gvar_ = sumall_except_dim<3>((out * broadcast<3>(slope_, in.shape_)) *
                        (temp_ - broadcast<3>(exp_, in.shape_)) *
                        -0.5f * F<power>(broadcast<3>(var_ + eps_, in.shape_), -1.5f));
      gexp_ = sumall_except_dim<3>(out * broadcast<3>(slope_, in.shape_));
      gexp_ *= -1.0f / F<square_root>(var_ + eps_);
      wtf_ = scale * sumall_except_dim<3>(-2.0f * (temp_ - broadcast<3>(exp_, in.shape_)));
      wtf_ *= gvar_;
      gexp_ += wtf_;
      gslope_ += sumall_except_dim<3>(out * in);
      gbias_ += sumall_except_dim<3>(out);
      in = (out * broadcast<3>(slope_, in.shape_)) *
           broadcast<3>(1.0f / F<square_root>(var_ + eps_), in.shape_) +
           broadcast<3>(gvar_, in.shape_) * scale * 2.0f * (temp_ - broadcast<3>(exp_, in.shape_)) +
           broadcast<3>(gexp_, in.shape_) * scale;

    }


  }

 private:
  mshadow::Random<xpu> *prnd_;
  int channel_;
  mshadow::Shape<4> in_shape_;
  mshadow::TensorContainer<xpu, 1> wtf_;
  mshadow::TensorContainer<xpu, 1> slope_;
  mshadow::TensorContainer<xpu, 1> gslope_;
  mshadow::TensorContainer<xpu, 1> bias_;
  mshadow::TensorContainer<xpu, 1> gbias_;
  mshadow::TensorContainer<xpu, 1> exp_;
  mshadow::TensorContainer<xpu, 1> gexp_;
  mshadow::TensorContainer<xpu, 1> var_;
  mshadow::TensorContainer<xpu, 1> gvar_;
  mshadow::TensorContainer<xpu, 1> running_exp_;
  mshadow::TensorContainer<xpu, 1> running_var_;
  mshadow::TensorContainer<xpu,4> temp_;
  float init_slope_;
  float init_bias_;
  float eps_;
  float bn_momentum_;
  Node<xpu> activations_;
  ILayer<xpu>* inputLayer_;

};  // class batchnorm_layer



#endif /* BATCHNORM_LAYER_H_ */
