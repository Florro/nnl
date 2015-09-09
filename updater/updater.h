/*
 * updater.h
 *
 *  Created on: Aug 23, 2015
 *      Author: niklas
 */

#ifndef UPDATER_H_
#define UPDATER_H_



#include "../layer/layer.h"
#include "UpdaterParameter.h"
#include "SGDUpdater.h"

namespace updater {


//AsyncUpdater class, contains specific updater(ex. SGD) and actual pserver interface
template<typename xpu>
class asyncUpdater{

public:

	 mshadow::ps::ISharedModel<xpu, real_t> *pserver;
	 mshadow::Tensor<xpu,2> weight;
	 mshadow::Tensor<xpu,2> grad;

	 asyncUpdater(int layer_index,
             int devid,
             int priority,
             mshadow::ps::ISharedModel<xpu, real_t> *pserver,
             mshadow::Tensor<xpu,2> weight,
             mshadow::Tensor<xpu,2> grad,
             IUpdater<xpu>* updater):
            	 updater_(updater), weight(weight), grad(grad), devid_(devid), pserver(pserver), data_key_(layer_index){ };



	 //init keys for pserver (call after constructing architecture)
	 void Init(){
		 pserver->InitKey(weight.shape_, data_key_, devid_);
		 updater_->Init();
	 }

	 //start syncing process of parameters (call after backprop)
	 void SyncProc(){
		 pserver->Push(grad, data_key_, devid_, -data_key_);
		 pserver->PullReq(grad, data_key_, devid_, -data_key_,
		 				      ApplyUpdate_, this);
	 }

	 //wait for syncing process to complete (call before forward)
	 void UpdateWait(){
		 pserver->PullWait(data_key_, devid_);
	 }

	 //sets current epoch for parameter scheduler
	 void set_epoch(int epoch){
		 this->updater_->set_epoch(epoch);
	 }

	 //sets hyperparams for updater
	 void set_params(UpdaterParam* hyperparam){
		 this->updater_->set_params(hyperparam);
	 }

	 //set streams for updaters
	 void set_stream(mshadow::Stream<xpu> *stream){
		 utility::Check(updater_ != NULL, "IUpdater not initialized correctly");
		 updater_->set_stream(stream);
	 }


	 //call back function, call itself(arg), static_cast for safety
	 inline static void ApplyUpdate_(mshadow::Stream<xpu> *stream, void *arg) {
		 asyncUpdater<xpu> *up = static_cast<asyncUpdater<xpu>*>(arg);
		 up->updater_->set_stream(stream);
		 up->updater_->Update();
	 }

private:

	int is_weight_;
    int devid_;
	int data_key_;
	IUpdater<xpu>* updater_;


};

//Creates local updater (ex. SGD)
template<typename xpu>
inline IUpdater<xpu>* CreateUpdater_(mshadow::Tensor<xpu,2> weight,
                                     mshadow::Tensor<xpu,2> wgrad,
                                     bool is_bias) {

  return new SGDUpdater<xpu>(weight, wgrad, is_bias);

}

//returns actual AsyncUpdater
template<typename xpu, int dim>
inline asyncUpdater<xpu>*
CreateAsyncUpdater_(int layer_index,
                    int devid,
                    int priority,
                    mshadow::ps::ISharedModel<xpu, real_t> *pserver,
                    mshadow::Tensor<xpu,dim> weight,
                    mshadow::Tensor<xpu,dim> wgrad) {

  return new asyncUpdater<xpu>(layer_index,
                               devid, priority,
                               pserver,
                               weight.FlatTo2D(), wgrad.FlatTo2D(),
                               CreateUpdater_(weight.FlatTo2D(), wgrad.FlatTo2D(), dim == 1));
}


//Creates a visitor, which creates correct updaters for each individual weight structure
template<typename xpu>
struct CreateAsyncUpdaterVisitor : public IVisitor<xpu> {

	  // layerid
	  int layerid;
	  // device id
	  int devid;
	  // parameter server
	  mshadow::ps::ISharedModel<xpu, real_t> *pserver;
	  // output updaters
	  std::vector< asyncUpdater<xpu>* > *out_updaters;

	  // constructor
	  CreateAsyncUpdaterVisitor(int layerid, int devid,  mshadow::ps::ISharedModel<xpu, real_t> *pserver, std::vector< asyncUpdater<xpu>*> *out_updaters)
									  : layerid(layerid), devid(devid),pserver(pserver), out_updaters(out_updaters) {}


	virtual void Visit( bool is_bias,
						mshadow::Tensor<xpu,1> weight,
	                    mshadow::Tensor<xpu,1> grad) {
		  out_updaters->push_back(CreateAsyncUpdater_(layerid + (int)is_bias, devid, -layerid, pserver, weight, grad));
		}
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,2> weight,
					   mshadow::Tensor<xpu,2> grad) {
	  	  out_updaters->push_back(CreateAsyncUpdater_(layerid + (int)is_bias, devid, -layerid, pserver, weight, grad));
  	  }
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,3> weight,
					   mshadow::Tensor<xpu,3> grad) {
	  	  out_updaters->push_back(CreateAsyncUpdater_(layerid + (int)is_bias, devid, -layerid, pserver, weight, grad));
  	  }
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,4> weight,
					   mshadow::Tensor<xpu,4> grad) {
	  	  out_updaters->push_back(CreateAsyncUpdater_(layerid + (int)is_bias, devid, -layerid, pserver, weight, grad));
  	  }

};

//Creates Asyncupdaters
template<typename xpu>
void CreateAsyncUpdaters(int layer_index,
                         int device_id,
                         mshadow::ps::ISharedModel<xpu, real_t> *param_server,
                         ILayer<xpu> *p_layer,
                         std::vector<asyncUpdater<xpu>*> *out_updaters){

	  CreateAsyncUpdaterVisitor<xpu> visitor(layer_index, device_id, param_server, out_updaters);
	  p_layer->ApplyVisitor(&visitor);
}

}


#endif /* UPDATER_H_ */
