/*
 * nntrainer.h
 *
 *  Created on: Aug 22, 2015
 *      Author: niklas
 */

#ifndef NNTRAINER_H_
#define NNTRAINER_H_

#include "mshadow/tensor.h"
#include "mshadow-ps/mshadow_ps.h"
#include "neural_net.h"
#include "../utility/dataBatchLoader_mthread.h"
#include "../io/iter.h"


// helper function to get the max index
inline int MaxIndex(Tensor<cpu, 1, real_t> pred) {
  int maxidx = 0;
  for (index_t i = 1; i < pred.size(0); ++i) {
    if(pred[i] > pred[maxidx]) maxidx = (int)i;
  }
  return maxidx;
}

template<typename xpu>
class nntrainer{
public:

	nntrainer(std::string net, std::vector < std::pair <std::string, std::string > > cfg):
		 	  net_(net), cfg_(cfg) {

		  std::vector<int> devices = configurator::getdevices(cfg);

		  ndev_ = devices.size();
		  for (int i = 0; i < devices.size(); ++i) {
			 devs_.push_back(devices[i]);
		  }

		  printf("Using %i devices\n", ndev_);

		  ps_ = mshadow::ps::CreateSharedModel<xpu, real_t>("local");
		  ps_->Init(devs_);

		  nets_.resize(ndev_);
		  for (int i = 0; i < ndev_; ++i) {
		     mshadow::InitTensorEngine<xpu>(devs_[i]);
		     nets_[i] = new ConvNet<xpu>(devs_[i], ps_);
		     nets_[i]->set_architecture(cfg);

		  }
		  nets_[0]->display_dim();

		  batch_size_ = configurator::getbatchsize(cfg_);
		  epochs_ = configurator::getepochs(cfg_);

		  utility::Check( (float(batch_size_) / float(ndev_) == (batch_size_ / ndev_) ), "Batchsize(%i) has to be divisible by ndev(%i)", batch_size_, ndev_);

		  start_epoch_ = configurator::getstartepoch(cfg);

	}

	void predict( ) {

		 //Create output dir
		  utility::createDir(net_, "output");

		  // load weights
		  if(start_epoch_ != 0){
			  for(int i = 0; i < ndev_; i++){
				  nets_[i]->load_weights(net_, start_epoch_);
				  nets_[i]->set_epoch(start_epoch_);
			  }
		  }
		  else{
			  utility::Error("Prediction with uninitialized net makes not much sense");
		  }


		  iter::IIter<iter::DataChunk> * testIterator = iter::CreateIter(net_, cfg_, false);
		  testIterator->Initalize();
		  testIterator->StartIter(start_epoch_);

		  double wall0 = utility::get_wall_time();


		  std::cout << "Test: ";

		  real_t nerr = 0;
		  real_t logloss = 0;
		  real_t fullsize = 0;
		  while ( testIterator->Next() ) {
			  this->predict_batch_(testIterator->Entry().Data, testIterator->Entry().Labels, nerr, logloss);
			  fullsize += testIterator->Entry().Size();
			  this->write_acts_(testIterator->Entry().Data);
		  }

		  //Print logloss/acc test and write to file
		  real_t acc = (1.0 - (real_t)nerr/fullsize)*100;
		  real_t loss = (-(real_t)logloss/fullsize);
		  printf("%.2f%% ", acc);
		  printf("logloss %.4f ", loss);

		  double wall1 = utility::get_wall_time();
		  std::cout << "( " << wall1 - wall0 << "s )" << std::endl;

		  delete testIterator;



	}


	void trainvalidate_batchwise() {

		  utility::createDir(net_, "log");
		  string logfile_ = net_ + "log/loss.log";


		  // load weights
		  if(start_epoch_ != 0){
			  for(int i = 0; i < ndev_; i++){
				  nets_[i]->load_weights(net_, start_epoch_);
			  }
		  }


		  int num_out = nets_[0]->get_outputdim();
		  int step = batch_size_ / ndev_;

		  iter::IIter<iter::DataChunk> * trainIterator = iter::CreateIter(net_, cfg_, true);
		  trainIterator->Initalize();

		  iter::IIter<iter::DataChunk> * testIterator = iter::CreateIter(net_, cfg_, false);
		  testIterator->Initalize();

		  //Epochs loop
		  for (int i = start_epoch_; i <= epochs_; ++ i){


			  real_t size = 0.0;
			  real_t train_nerr = 0;
			  real_t train_logloss = 0;
			  double wall0 = utility::get_wall_time();

			  //trainDataLoader.start_epoch(i);
			  trainIterator->StartIter(i);
			  testIterator->StartIter(i);


			  while ( trainIterator->Next() ) {

				  if ( trainIterator->Entry().Size() < batch_size_ ) continue;

				 // running parallel threads
				  #pragma omp parallel num_threads(ndev_) reduction(+:train_nerr,train_logloss)
				  {
					int tid = omp_get_thread_num();
					mshadow::SetDevice<xpu>(devs_[tid]);

					// temp output layer
					TensorContainer<cpu, 2, real_t> pred;
					pred.Resize(Shape2(step, num_out));
					  //set epoch for updater
					  nets_[tid]->set_epoch(i);
					  // run forward
					  nets_[tid]->Forward(trainIterator->Entry().Data.Slice(tid * step, (tid + 1) * step), pred, true);
					  // run backprop
					  nets_[tid]->Backprop(&trainIterator->Entry().Labels[tid * step]);
					  //evaluate prediction
					  this->eval_pred_(pred, &trainIterator->Entry().Labels[tid * step], train_nerr, train_logloss);
				  }
				  size += trainIterator->Entry().Size();
			  }



			  real_t acc_train = (1.0 - (real_t)train_nerr/size)*100;
			  real_t log_train = -(real_t)train_logloss/size;
			  printf("Epoch: %i, Train: ", i);
			  printf("%.2f%% ", acc_train);
			  printf("logloss %.4f\n", log_train);
			  utility::write_val_to_file< float >(logfile_.c_str(), acc_train, false);
			  utility::write_val_to_file< float >(logfile_.c_str(), log_train, false);



			  std::cout << "Test: ";

			  real_t nerr = 0;
			  real_t logloss = 0;
			  real_t fullsize = 0;
			  while ( testIterator->Next() ) {
				  this->predict_batch_(testIterator->Entry().Data, testIterator->Entry().Labels, nerr, logloss);
				  fullsize += testIterator->Entry().Size();
			  }

			  //Print logloss/acc test and write to file
			  real_t acc = (1.0 - (real_t)nerr/fullsize)*100;
			  real_t loss = (-(real_t)logloss/fullsize);
			  printf("%.2f%% ", acc);
			  printf("logloss %.4f ", loss);
			  utility::write_val_to_file< float >(logfile_.c_str(), acc, false);
			  utility::write_val_to_file< float >(logfile_.c_str(), loss, true);


			  if(((i == epochs_) or (i % 25 == 0) ) and (i != 0)){
				  nets_[0]->Sync();
				  nets_[0]->save_weights(net_, i);
			  }


			  double wall1 = utility::get_wall_time();
			  std::cout << "( " << wall1 - wall0 << "s )" << std::endl;

		  }

		  delete trainIterator;
		  delete testIterator;

	}



	void save_weights(){
		 nets_[0]->Sync();
	     nets_[0]->save_weights(net_, 0);
	}


	~nntrainer(){
	 for(int i = 0; i < ndev_; ++i) {
		  mshadow::SetDevice<xpu>(devs_[i]);
		  delete nets_[i];
		  ShutdownTensorEngine<xpu>();
	  }
	}

private:

	mshadow::ps::ISharedModel<xpu, real_t> *ps_;
	int ndev_;
	std::string net_;
	std::vector<int> devs_;
	std::vector<INNet *> nets_;
    std::vector < std::pair <std::string, std::string > > cfg_;
    int batch_size_;
    int epochs_;
    int start_epoch_;

    void eval_pred_(TensorContainer<cpu, 2, real_t> &pred, int* ytest, real_t & ext_nerr, real_t & ext_logloss){
    	 for (int k = 0; k < pred.size(0); ++ k) {
    		 ext_nerr   += MaxIndex(pred[k]) != *(ytest + k);
    		 ext_logloss += save_log(pred[ k ][*(ytest + k)]);
		 }
    }


    void predict_batch_(TensorContainer<cpu, 4, real_t> &xtest, std::vector<int> &ytest, real_t & ext_nerr, real_t & ext_logloss){
		// mini-batch per device
		int num_out = nets_[0]->get_outputdim();
		  // evaluation
		  real_t nerr = 0;
		  real_t logloss = 0;

		  if ( xtest.size(0) == batch_size_ ) {

			int step = batch_size_ / ndev_;

			#pragma omp parallel num_threads(ndev_) reduction(+:nerr,logloss)
			{
			  int tid = omp_get_thread_num();
			  mshadow::SetDevice<xpu>(devs_[tid]);

			  // temp output layer
			  TensorContainer<cpu, 2, real_t> pred;
			  pred.Resize(Shape2(step, num_out));

			  nets_[tid]->Forward(xtest.Slice(tid * step, (tid + 1) * step), pred, false);
			  for (int k = 0; k < step; ++ k) {
				nerr   += MaxIndex(pred[k]) != ytest[tid * step + k];
				logloss += (save_log(pred[ k ][ytest[tid * step + k]]));
			  }
			}
		  }
		  else {
			  // temp output layer
			  TensorContainer<cpu, 2, real_t> pred;
			  pred.Resize(Shape2(xtest.size(0), num_out));
			  nets_[0]->Forward(xtest, pred, false);
			  for (int k = 0; k < xtest.size(0); ++ k) {
				nerr   += MaxIndex(pred[k]) != ytest[k];
				logloss += (save_log(pred[k][ytest[k]]));
			  }

		  }

		  ext_nerr += nerr;
		  ext_logloss += logloss;
	}

    void write_acts_(TensorContainer<cpu, 4, real_t> &xtest){
		// mini-batch per device
		int step = batch_size_ / ndev_;
		int num_out = nets_[0]->get_outputdim();

		if ( xtest.size(0) == batch_size_ ) {
			  mshadow::SetDevice<xpu>(devs_[0]);
			  // temp output layer
			  TensorContainer<cpu, 2, real_t> pred;
			  pred.Resize(Shape2(step, num_out));
			  for(int i = 0; i < ndev_; i++){
				  nets_[0]->Forward(xtest.Slice(i * step, (i + 1) * step), pred, false);
				  nets_[0]->save_activations(nets_[0]->get_arch_size()-1, net_ + "output/predictions.csv");
				  nets_[0]->save_activations(nets_[0]->get_arch_size()-3, net_ + "output/lastlayer.csv");
			  }
		}
		else {
			  //HARDFUCK potential memory peak
			  // temp output layer
			  TensorContainer<cpu, 2, real_t> pred;
			  pred.Resize(Shape2(xtest.size(0), num_out));
			  nets_[0]->Forward(xtest, pred, false);
			  nets_[0]->save_activations(nets_[0]->get_arch_size()-1, net_ + "output/predictions.csv");
			  nets_[0]->save_activations(nets_[0]->get_arch_size()-3, net_ + "output/lastlayer.csv");
		}
    }












};


#endif /* NNTRAINER_H_ */
