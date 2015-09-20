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
#include "../utility/dataBatchLoader.h"

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

	nntrainer(int argc, char *argv[],  std::string net, std::vector < std::pair <std::string, std::string > > cfg):
		 	  net_(net), cfg_(cfg) {

		  utility::createDir(net, "log");
		  logfile_ = net + "log/loss.log";

		  ndev_ = argc - 2;
		  for (int i = 2; i < argc; ++i) {
			 devs_.push_back(atoi(argv[i]));
		  }

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
	}


	void trainvalidate_batchwise( const std::string & train_path , const std::string & test_path, bool augment_data, unsigned junkSize) {

		  // mini-batch per device
		  int num_out = nets_[0]->get_outputdim();
		  int step = batch_size_ / ndev_;

		  // Create Batch-loaders for Data with max Junksize and shuffle
		  dataload::dataBatchLoader trainDataLoader(junkSize, true, augment_data, cfg_);
		  dataload::dataBatchLoader testDataLoader(junkSize, false, false, cfg_);
		  std::cout << std::endl << std::endl;


		  //Epochs loop
		  for (int i = 0; i <= epochs_; ++ i){

			  int b = 1;
			  while ( !trainDataLoader.finished() ) {

				  // Load databatch from disk
				  trainDataLoader.readBatch();

					  // running parallel threads
				  #pragma omp parallel num_threads(ndev_)
				  {
					int tid = omp_get_thread_num();
					mshadow::SetDevice<xpu>(devs_[tid]);

					// temp output layer
					TensorContainer<cpu, 2, real_t> pred;
					pred.Resize(Shape2(step, num_out));

					for (index_t j = 0; j + batch_size_ <= trainDataLoader.Data().size(0); j += batch_size_) {
					  //set epoch for updater
					  nets_[tid]->set_epoch(i);
					  // run forward
					  nets_[tid]->Forward(trainDataLoader.Data().Slice(j + tid * step, j + (tid + 1) * step), pred, true);
					  // run backprop
					  nets_[tid]->Backprop(&trainDataLoader.Labels()[j + tid * step]);
					}
				  }

				  // evaluation
				  printf("Epoch: %i, Masterbatch: %u/%u, Train: ", i, b, trainDataLoader.numBatches());
				  long train_nerr = 0;
				  long train_logloss = 0;
				  this->predict_batch_(trainDataLoader.Data(), trainDataLoader.Labels(), train_nerr, train_logloss);
				  printf("%.2f%% ", (1.0 - (real_t)train_nerr/trainDataLoader.Data().size(0))*100);
				  printf("logloss %.4f\n", (-(real_t)train_logloss/trainDataLoader.Data().size(0)));
				  b++;

			  }
			  //reset data loader
			  trainDataLoader.reset();


			  //Cout logging
			  std::cout << "Test: ";

			  long nerr = 0;
			  long logloss = 0;
			  while ( !testDataLoader.finished() ) {
				  testDataLoader.readBatch();
				  this->predict_batch_(testDataLoader.Data(), testDataLoader.Labels(), nerr, logloss);
			  }
			  printf("%.2f%% ", (1.0 - (real_t)nerr/testDataLoader.fullSize())*100);
			  printf("logloss %.4f\n", (-(real_t)logloss/testDataLoader.fullSize()));
			  utility::write_val_to_file< float >(logfile_.c_str(), -(real_t)logloss/testDataLoader.fullSize());
			  testDataLoader.reset();

			  if(i == epochs_){
				  nets_[0]->Sync();
				  nets_[0]->save_weights(net_, i);
			  }

		  }

		}


	void predict( const std::string & test_path, unsigned junkSize, unsigned epoch) {

		  // mini-batch per device
		  for(int i = 0; i < ndev_; i++){
			  nets_[0]->load_weights(net_, epoch);
			  nets_[1]->load_weights(net_, epoch);
		  }

		  // Create Batch-loaders for Data with max Junksize and shuffle
		  dataload::dataBatchLoader testDataLoader(junkSize, false, false, cfg_);
		  std::cout << std::endl << std::endl;

		  //Cout logging
		  std::cout << "Test: ";

		  long nerr = 0;
		  long logloss = 0;
		  while ( !testDataLoader.finished() ) {
			  testDataLoader.readBatch();
			  this->predict_batch_(testDataLoader.Data(), testDataLoader.Labels(), nerr, logloss);

		  }
		  printf("%.2f%% ", (1.0 - (real_t)nerr/testDataLoader.fullSize())*100);
		  printf("logloss %.4f\n", (-(real_t)logloss/testDataLoader.fullSize()));
		  utility::write_val_to_file< float >(logfile_.c_str(), -(real_t)logloss/testDataLoader.fullSize());
		  testDataLoader.reset();

		  //save acts and current weights.
		  std::string holdoutfile = net_ + "output/holdout_";
		  this->write_acts(testDataLoader.Data(), holdoutfile);



		}


	void write_acts(TensorContainer<cpu, 4, real_t> &xtest, std::string outputfile){
		  utility::createDir(net_, "output");
		  // mini-batch per device
		  int step = batch_size_ / ndev_;
		  int num_out = nets_[0]->get_outputdim();

		  // evaluation
		  mshadow::SetDevice<xpu>(devs_[0]);
		  TensorContainer<cpu, 2, real_t> pred;
		  pred.Resize(Shape2(step, num_out));

		  for (index_t j = 0; j + step <= xtest.size(0); j += step) {
				nets_[0]->Forward(xtest.Slice(j, j + step), pred, false);
				//Save activations
				nets_[0]->save_activations(nets_[0]->get_arch_size()-1, outputfile+"predictions.csv");
				nets_[0]->save_activations(nets_[0]->get_arch_size()-3, outputfile+"lastlayer_activations.csv");
		  }

		  //predict last bit of data
		  int lastbit = xtest.size(0) % step;
		  int laststart = xtest.size(0) / step;
		  int lastsize = xtest.size(0) - laststart*step ;
		  if(lastbit != 0){
			  // temp output layer
			  TensorContainer<cpu, 2, real_t> pred;
			  pred.Resize(Shape2(lastsize, num_out));
			  nets_[0]->Forward(xtest.Slice(laststart*step, xtest.size(0)), pred, false);
			  nets_[0]->save_activations(nets_[0]->get_arch_size()-1, outputfile+"predictions.csv");
			  nets_[0]->save_activations(nets_[0]->get_arch_size()-3, outputfile+"lastlayer_activations.csv");
		  }

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
	TensorContainer<cpu, 4, real_t> xtrain_augmented_;
    std::string logfile_;
    std::vector < std::pair <std::string, std::string > > cfg_;
    int batch_size_;
    int epochs_;

    void predict_batch_(TensorContainer<cpu, 4, real_t> &xtest, std::vector<int> &ytest, long & ext_nerr, long & ext_logloss){
		// mini-batch per device
		  int num_out = nets_[0]->get_outputdim();
		  int step = batch_size_ / ndev_;

		  // evaluation
		  long nerr = 0;
		  long logloss = 0;


		  #pragma omp parallel num_threads(ndev_) reduction(+:nerr,logloss)
		  {
			int tid = omp_get_thread_num();
			mshadow::SetDevice<xpu>(devs_[tid]);

			// temp output layer
			TensorContainer<cpu, 2, real_t> pred;
			pred.Resize(Shape2(step, num_out));

			for (index_t j = 0; j + batch_size_ <= xtest.size(0); j += batch_size_) {
			  nets_[tid]->Forward(xtest.Slice(j + tid * step, j + (tid + 1) * step), pred, false);
			  for (int k = 0; k < step; ++ k) {
				nerr   += MaxIndex(pred[k]) != ytest[j + tid * step + k];
				logloss += (save_log(pred[ k ][ytest[j + tid * step + k]]));
			  }
			}
		  }

		  //predict last bit of data
		  int lastbit = xtest.size(0) % batch_size_;
		  int laststart = xtest.size(0) / batch_size_;
		  int lastsize = xtest.size(0) - laststart*batch_size_ ;
		  if(lastbit != 0){
			  // temp output layer
			  TensorContainer<cpu, 2, real_t> pred;
			  pred.Resize(Shape2(lastsize, num_out));
			  nets_[0]->Forward(xtest.Slice(laststart*batch_size_, xtest.size(0)), pred, false);
			  for (int k = 0; k < lastsize; ++ k) {
				nerr   += MaxIndex(pred[k]) != ytest[laststart + k];
				logloss += (save_log(pred[ k ][ytest[laststart + k]]));
			  }
		  }



		  ext_nerr += nerr;
		  ext_logloss += logloss;
	}







};


#endif /* NNTRAINER_H_ */
