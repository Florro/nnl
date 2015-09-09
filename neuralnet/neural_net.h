/*
 * neural_net.h
 *
 *  Created on: Aug 20, 2015
 *      Author: niklas
 */

#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <iostream>
#include <vector>

#include "mshadow/tensor.h"
#include "mshadow-ps/mshadow_ps.h"
#include "mshadow/io.h"


#include "../layer/layer.h"
#include "../updater/updater.h"
#include "configurator.h"
#include "../updater/UpdaterParameter.h"
#include "../io/modelstate.h"

// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

/*! interface for nnet, interface allows use to use GPU/CPU implementation in a unified way */
class INNet{
 public:
  virtual void Forward(const Tensor<cpu, 4, real_t>& inbatch, Tensor<cpu, 2, real_t> &oubatch, bool is_train) = 0;
  virtual void Backprop(int* labels) = 0;
  virtual void display_dim() = 0;
  virtual void set_architecture(std::string configfile) = 0;
  virtual void set_batchSize( int batch_size ) = 0;
  virtual void set_epoch(int epoch) = 0;
  virtual int get_max_epoch() = 0;
  virtual void save_activations(int n_layer, std::string outputfile ) = 0;
  virtual int get_outputdim() = 0;

  virtual void Sync() = 0;
  virtual void save_weights(std::string outputfile) = 0;
  virtual ~INNet(){};
};

/*! Main NeuralNet thread */
template<typename xpu>
class ConvNet : public INNet {
 public:
  // initialize the network
  ConvNet(int devid, mshadow::ps::ISharedModel<xpu, real_t> *p_server)
      :rnd_(0), devid_(devid), pserver_(p_server), batchSize_(0), losslayer_(NULL) {

	//setup Device
	mshadow::SetDevice<xpu>(devid_);
    // setup stream
    stream_ = mshadow::NewStream<xpu>();

  }

  virtual ~ConvNet() {

	  //Cleanup
	  mshadow::SetDevice<xpu>(devid_);
	  mshadow::DeleteStream(stream_);

  }


  virtual void set_architecture(std::string configfile){

	    ConfigIterator* myreader = new ConfigIterator(configfile.c_str());
		//auto_ptr<ConfigIterator> myreader(new ConfigIterator(configfile.c_str()));
		std::vector < std::pair <std::string, std::string > > cfg;
		while(!myreader->isEnd()){
			myreader->Next();
			cfg.push_back(std::make_pair(myreader->name(), myreader->val()));
		}

	    hyperparam_ = new updater::UpdaterParam();

		//Read config and generate network architecture
	    readNetConfig(cfg, architecture_, losslayer_, hyperparam_, stream_,  rnd_);

	    //disable backpropagation of error into data layer
	    architecture_[1]->setBackpropError(false);


	    //init async Updater class
	    /*
	     * every layer gets n updaters, with n number of parameter structures
	     * ex.
	     * standardlayer: 2 updaters (bias, weights)
	     * datalayer: 0 updaters
	     *
	     */
	    for(int i = 0; i < architecture_.size(); i++){
	    	std::vector< updater::asyncUpdater<xpu>* > temp_updaters;
	    	//Creates Asyncupdaters
	    	updater::CreateAsyncUpdaters<xpu>(2*i,   devid_,  pserver_, architecture_[i], &temp_updaters);
	    	//Init Asyncupdaters
	    	for(int j = 0; j < temp_updaters.size(); j++){
	    		temp_updaters[j]->set_stream(stream_);
	    		temp_updaters[j]->Init();
	    		temp_updaters[j]->set_params(hyperparam_);
	    	}
	    	async_updaters_.push_back(temp_updaters);
	    }
  }

  //Adjust layers to new batchsize
  virtual void set_batchSize( int batch_size ){
	  batchSize_ = batch_size;
	  for(unsigned i = 0; i < architecture_.size(); i++){
		architecture_[i]->onBatchSizeChanged( batch_size );
	  }
  }

  //Perform pseudo update for sync purposes
  virtual void Sync(){
	for(unsigned i = 0; i < architecture_.size(); i++){
		//Wait until syncing with server completed
		for(unsigned j = 0; j < async_updaters_[i].size(); j++){
			async_updaters_[i][j]->UpdateWait();
		}
	}
  }

  // forward propagation
  virtual void Forward(const Tensor<cpu, 4, real_t>& inbatch, Tensor<cpu, 2, real_t> &oubatch, bool is_train) {

	//Check if batchsize has changed, adjust if necessary
	if(inbatch.size(0) != batchSize_)
		this->set_batchSize(inbatch.size(0));

    // copy data to input layer
	Copy(architecture_[0]->getpAct()->data, inbatch, stream_);

	//Perform forward pass
    for(unsigned i = 0; i < architecture_.size(); i++){
    	//Wait until syncing with server completed
    	for(unsigned j = 0; j < async_updaters_[i].size(); j++){
    		async_updaters_[i][j]->UpdateWait();
    	}
    	//feed forward
	  	architecture_[i]->feedforward(is_train);
    }

    // loss calculation
    losslayer_->feedforward();

    // copy result out
    Copy(oubatch, architecture_.back()->getpAct()->mat(), stream_);
    stream_->Wait();

  }

  // back propagation
  virtual void Backprop(int* labels) {

	//Calculated gradient in losslayer_
	losslayer_->backpropagate(labels);

	//Perform backward pass
	for(int i = architecture_.size()-1; i > 0; i--){
	  architecture_[i]->backpropagate();
	  // wait backprop to complete
	  if (async_updaters_[i].size() != 0) stream_->Wait();
	  //Start syncing of parameters (ASYNC)
	  for (unsigned j = 0; j < async_updaters_[i].size(); ++j) {
		  async_updaters_[i][j]->SyncProc();
	  }
	}
  }

  //set current epoch for parameter scheduler
  virtual void set_epoch(int epoch){
	  for(unsigned i = 0; i < async_updaters_.size(); i++){
		  for (unsigned j = 0; j < async_updaters_[i].size(); ++j) {
			  async_updaters_[i][j]->set_epoch(epoch);
		  }
	  }
  }


  //display network architecture
  virtual void display_dim(){

	  	int neurons = 0;
	  	int synapses = 0;
	    std::cout << "Layer Dimensions: " << std::endl << std::endl;

		for (unsigned i = 0; i < this->architecture_.size(); i++){
			std::cout 	<< i << ".\t"
						<< this->architecture_[i]->getType() << "\t"
						<< this->architecture_[i]->getpAct()->data.size(1) << " x "
						<< this->architecture_[i]->getpAct()->data.size(2) << " x "
						<< this->architecture_[i]->getpAct()->data.size(3)
						<< std::endl;

			neurons += this->architecture_[i]->getpAct()->data.size(0) * this->architecture_[i]->getpAct()->data.size(1) * this->architecture_[i]->getpAct()->data.size(2) * this->architecture_[i]->getpAct()->data.size(3);
			synapses += this->architecture_[i]->getParamSize();
		}

		std::cout << std::endl;
		std::cout << "Losstype: " << losslayer_->getType() << std::endl;

		std::cout << std::endl;
		printf("#Neurons:  %i\t( %.2f MB )\n", neurons, (neurons*sizeof(real_t)/real_t(1024 * 1024)) );
		printf("#Synapses: %i\t( %.2f MB )\n\n", synapses, (synapses*sizeof(real_t) /real_t(1024 * 1024)) );


	}

  //return number of labels
  virtual int get_outputdim(){
	  return this->architecture_.back()->getpAct()->data.size(3);
  }
  //return number of max epochs
  virtual int get_max_epoch(){
	  return this->hyperparam_->epochs;
  }

  //save activations in layer n_layer to file
  virtual void save_activations(int n_layer, std::string outputfile ){

	  TensorContainer<cpu, 2> host_acts(architecture_[n_layer]->getpAct()->mat().shape_);
	  Copy(host_acts, architecture_[n_layer]->getpAct()->mat(), stream_);

	   //Write params into file
	  std::ofstream outputstream ((char*)outputfile.c_str(), std::fstream::app);


	  if (outputstream.is_open()){
	  for (int p = 0; p < host_acts.size(0); p++){
		  for(int i = 0; i < host_acts.size(1); i++){
			  outputstream << host_acts[p][i];
			  if(i % (host_acts.size(1)) != (host_acts.size(1)-1)){
				  outputstream << ",";
			  }
		  }
		  outputstream << std::endl;
      }
	  outputstream.close();
	  }
  }

  //save activations in layer n_layer to file
  virtual void save_weights(std::string outputfile ){

	  for(int i = 0; i < architecture_.size(); i++){
		  modelstate::save_weights(architecture_[i],i);
	  }


  }


 private:

  //batchsize
  int batchSize_;

  // random seed generator
  Random<xpu, real_t> rnd_;

  // device id
  int devid_;
  // parameter server interface
  mshadow::ps::ISharedModel<xpu, real_t> *pserver_;

  // computing stream
  mshadow::Stream<xpu> *stream_;

  //architecture_ carries layers
  std::vector< ILayer<xpu>* > architecture_;
  Loss_layer<xpu>* losslayer_;

  //Parameterserver interface
  std::vector<std::vector< updater::asyncUpdater<xpu>*> > async_updaters_;

  //Updater parameter
  updater::UpdaterParam* hyperparam_;

};


namespace mshadow {
	namespace ps {
		// model updater is used when update is happening on server side
		// if we only use parameter server for sum aggregation
		// this is not needed, but we must declare this function to return NULL
		template<>
		IModelUpdater<real_t> *CreateModelUpdater(void) {
		  return NULL;
		}
	}
}


#endif /* NEURAL_NET_H_ */
