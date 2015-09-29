/*
 * UpdaterParameter.h
 *
 *  Created on: Aug 22, 2015
 *      Author: niklas
 */

#ifndef UPDATERPARAMETER_H_
#define UPDATERPARAMETER_H_

namespace updater{

/*! \brief potential parameters for each layer */
struct UpdaterParam {

   /* \brief momentum */
   float momentum;
   /*! \brief learning rate */
   float learning_rate;
   /*! \brief weight decay */
   float weightdecay;


   // scheduling parameters learning rate
   int lr_schedule;
   /*! \brief base learning rate */
   float base_lr_;
   /*! \brief decay parameter gamma */
   float lr_decay;
   /*! \brief minimum learning rate */
   float lr_minimum;


   // scheduling parameters momentum
   int momentum_schedule;
   /*! \brief final momentum */
   float final_momentum_;
   /*! \brief base momentum */
   float base_momentum_;
   /*! \brief saturation epoch for momentum */
   int saturation_epoch_;

   /*! \brief epochs  */
   int max_epochs;

   /*! \brief batchsize  */
   int batchsize;


  /*! \brief clip gradient (remove nans, clip gradient at +- clip) */
  float clipgradient;
  /*! \brief constructor that sets default parameters */
  UpdaterParam(void) {

	lr_schedule = 1;
	momentum_schedule = 1;

	base_lr_ = 0.01;
    lr_decay = 0.97f;
    lr_minimum = 0.0001f;

    base_momentum_ = 0.5f;
    final_momentum_ = 0.9f;
    saturation_epoch_ = 10;

    weightdecay = 0.00001f;
    clipgradient = 0.0f;
    max_epochs = 10;

    batchsize = 100;

  }

  /*! \brief learning parameter scheduler */
  inline void ScheduleParams(int epoch) {
    switch (lr_schedule) {
      case 0: learning_rate = base_lr_; break;
      case 1: learning_rate = base_lr_ * powf(lr_decay, float(epoch)); break;
      default: utility::Error("Unknown learning rate schedule!");
    }

    //lr schedule
    learning_rate = learning_rate < lr_minimum ? lr_minimum : learning_rate;

    //momentum schedule
    if (momentum_schedule) {
	  momentum = epoch * (final_momentum_ - base_momentum_) / saturation_epoch_ + base_momentum_;
	}
	momentum = momentum < final_momentum_ ? momentum : final_momentum_;

  }





};

}

#endif /* UPDATERPARAMETER_H_ */
