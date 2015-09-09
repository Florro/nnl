/*
 * SGDUpdater.h
 *
 *  Created on: Aug 23, 2015
 *      Author: niklas
 */

#ifndef SGDUPDATER_H_
#define SGDUPDATER_H_

#include "UpdaterParameter.h"
#include "mshadow/tensor.h"
#include "../utility/mshadow_op.h"

namespace updater{

//Base class for updaters
template<typename xpu>
class IUpdater {

public:
	virtual void Update() = 0;
	virtual void Init() = 0;
	virtual void set_stream(mshadow::Stream<xpu>*) = 0;
	virtual void set_epoch(int) = 0;
	virtual void set_params(UpdaterParam*) = 0;
	virtual ~IUpdater(){};
};



// Stochastic Gradient Descent data structure
template<typename xpu>
class SGDUpdater : public IUpdater<xpu> {

public:
  mshadow::Tensor<xpu, 2> weight, wgrad;
  mshadow::TensorContainer<xpu, 2> momentum_gradient;
  UpdaterParam* hyperparams_;
  int epoch_;
  bool is_bias;

  // constructor
  SGDUpdater(mshadow::Tensor<xpu, 2> weight,
			  mshadow::Tensor<xpu, 2> grad,
			  bool is_bias)
	  : weight(weight), wgrad(grad),
		is_bias(is_bias) {}

  //applies update on parameters
  void Update(){
	  this->ApplyUpdate_(wgrad);
	  wgrad = 0.0f;
  }

  //init function
  void Init(){
	  momentum_gradient.Resize(weight.shape_, 0.0f);
  }

  void set_params(UpdaterParam* hyperparams){
	  hyperparams_ = hyperparams;
  }

  //set correct stream
  void set_stream(mshadow::Stream<xpu> *stream){
	    momentum_gradient.set_stream(stream);
		weight.set_stream(stream);
		wgrad.set_stream(stream);
  }

  //update epoch for parameter scheduler
  void set_epoch(int epoch){
	  this->epoch_ = epoch;
  }

private:

    //actual updater routine
  	inline void ApplyUpdate_(mshadow::Tensor<xpu, 2> grad) {

		hyperparams_->ScheduleParams(epoch_);
		if(hyperparams_->clipgradient != 0.0f){
			if (!is_bias) {
				momentum_gradient *= hyperparams_->momentum;
				momentum_gradient += (-hyperparams_->learning_rate) * (hyperparams_->weightdecay * weight + F<clip>(grad, hyperparams_->clipgradient));
			} else {
				momentum_gradient = (-hyperparams_->learning_rate) * F<clip>(grad, hyperparams_->clipgradient);
			}
		}
		else{
			if (!is_bias) {
				momentum_gradient *= hyperparams_->momentum;
				momentum_gradient += (-hyperparams_->learning_rate) * (hyperparams_->weightdecay * weight + grad);
			} else {
				momentum_gradient = (-hyperparams_->learning_rate) * grad;
			}
		}

		weight += momentum_gradient;

	}

};

}

#endif /* SGDUPDATER_H_ */
