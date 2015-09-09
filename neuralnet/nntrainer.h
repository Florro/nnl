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
#include "../utility/image_augmenter.h"
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

	nntrainer(int argc, char *argv[], std::string net):
		  myIA_(NULL), logfile_(net + "/loss.log"), net_(net) {

		  ndev_ = argc - 2;
		  // choose which version to use
		  for (int i = 2; i < argc; ++i) {
			 devs_.push_back(atoi(argv[i]));
		  }

		  ps_ = mshadow::ps::CreateSharedModel<xpu, real_t>("local");
		  ps_->Init(devs_);

		  nets_.resize(ndev_);
		  for (int i = 0; i < ndev_; ++i) {
		     mshadow::InitTensorEngine<xpu>(devs_[i]);
		     nets_[i] = new ConvNet<xpu>(devs_[i], ps_);
		     nets_[i]->set_architecture(net_ + "/config.conf");

		  }
		  nets_[0]->display_dim();
	}

	void trainvalidate( TensorContainer<cpu, 4, real_t> &xtrain,  TensorContainer<cpu, 4, real_t> &xtest,
								std::vector<int> &ytrain, std::vector<int> &ytest, bool augment_data){
	 // mini-batch per device
	  int batch_size = 100;
	  int num_out = nets_[0]->get_outputdim();
	  int epochs = nets_[0]->get_max_epoch();

	  int step = batch_size / ndev_;

	  if(augment_data){
		  myIA_ = new ImageAugmenter();
	  }
	  xtrain_augmented_.Resize(xtrain.shape_);


	  for (int i = 0; i < epochs; ++ i){
		  if(augment_data){
		  std::cout << "aug ";
			  for(int a = 0; a < xtrain.size(0); a++){
				  Copy(xtrain_augmented_[a], myIA_->distort_img(xtrain[a]));
				  //myIA_->display_img(xtrain_augmented_[a]);
				  //myIA_->compare_img(xtrain[a], xtrain_augmented_[a]);
			  }
		  }else{
			  xtrain_augmented_ = xtrain;
		  }


		  // running parallel threads
		  #pragma omp parallel num_threads(ndev_)
		  {
			int tid = omp_get_thread_num();
			mshadow::SetDevice<xpu>(devs_[tid]);

			// temp output layer
			TensorContainer<cpu, 2, real_t> pred;
			pred.Resize(Shape2(step, num_out));


			for (index_t j = 0; j + batch_size <= xtrain_augmented_.size(0); j += batch_size) {
			  //set epoch for updater
		      nets_[tid]->set_epoch(i);
			  // run forward
			  nets_[tid]->Forward(xtrain_augmented_.Slice(j + tid * step, j + (tid + 1) * step), pred, true);
			  // run backprop
			  nets_[tid]->Backprop(&ytrain[j + tid * step]);
			}
		  }


		  /*
		  if(i == 60 || i == 90 || i == 120 || i == 149){
			  std::string holdoutfile = "/home/niklas/Desktop/retina_data_3class/acts/" + utility::custom_to_string(i) + "_holdout_";
			  this->write_acts(xtest, holdoutfile);
			  std::string testfile = "/home/niklas/Desktop/retina_data_3class/acts/" + utility::custom_to_string(i) + "_train_";
			  this->write_acts(xtrain, testfile);
		  }
		  */

		  // evaluation
		  std::cout << "Epoch " << i << " Train: ";
		  this->predict(xtrain,ytrain);
		  std::cout << "Test: ";
		  this->predict(xtest,ytest);
		  std::cout << std::endl;

	  }

	}


	void trainvalidate_batchwise( const std::string & train_path , const std::string & test_path, bool augment_data) {

		  // mini-batch per device
		  int batch_size = 100;
		  int num_out = nets_[0]->get_outputdim();
		  int epochs = nets_[0]->get_max_epoch();

		  int step = batch_size / ndev_;

		  if(augment_data){
			  myIA_ = new ImageAugmenter();
		  }

		  // Create Batch-loaders for train and testdata
		  dataBatchLoader trainDataLoader(train_path, 10000, true);
		  dataBatchLoader testDataLoader(test_path, 10000, false);

		  for (int i = 0; i <= epochs; ++ i){

			  int b = 1;
			  while ( !trainDataLoader.finished() ) {

				  // Load databatch from disk
				  trainDataLoader.readBatch();

				  xtrain_augmented_.Resize(trainDataLoader.X().shape_);

				  if(augment_data){
				  std::cout << "aug ";
					  for(int a = 0; a < trainDataLoader.X().size(0); a++){
						  Copy(xtrain_augmented_[a], myIA_->distort_img(trainDataLoader.X()[a]));
						  //myIA_->compare_img(xtrain[a], xtrain_augmented_[a]);
					  }
				  }else{
					  xtrain_augmented_ = trainDataLoader.X();
				  }

				  // running parallel threads
				  #pragma omp parallel num_threads(ndev_)
				  {
					int tid = omp_get_thread_num();
					mshadow::SetDevice<xpu>(devs_[tid]);

					// temp output layer
					TensorContainer<cpu, 2, real_t> pred;
					pred.Resize(Shape2(step, num_out));

					for (index_t j = 0; j + batch_size <= xtrain_augmented_.size(0); j += batch_size) {
					  //set epoch for updater
					  nets_[tid]->set_epoch(i);
					  // run forward
					  nets_[tid]->Forward(xtrain_augmented_.Slice(j + tid * step, j + (tid + 1) * step), pred, true);
					  // run backprop
					  nets_[tid]->Backprop(&trainDataLoader.y()[j + tid * step]);
					}
				  }

				  // evaluation
				  printf("Epoch: %i, Masterbatch: %u/%u, Train: ", i, b, trainDataLoader.numBatches());
				  //std::cout << "Epoch " << i << " Masterbatch: " << b << " Train: ";
				  this->predict(trainDataLoader.X(),trainDataLoader.y());

				  printf("\n");
				  b++;

			  }
			  trainDataLoader.reset();

			  std::cout << "Test: ";

			  long nerr = 0.0;
			  long logloss = 0.0;
			  while ( !testDataLoader.finished() ) {
				  testDataLoader.readBatch();
				  this->predict_batch(testDataLoader.X(), testDataLoader.y(), nerr, logloss);
			  }
			  testDataLoader.reset();
			  printf("%.2f%% ", (1.0 - (real_t)nerr/testDataLoader.fullSize())*100);
			  printf("logloss %.4f\n", (-(real_t)logloss/testDataLoader.fullSize()));
			  utility::write_val_to_file< float >(logfile_.c_str(), -(real_t)logloss/testDataLoader.fullSize());


			  if(i == epochs){
				  nets_[0]->Sync();
				  nets_[0]->save_weights("");
				  std::string holdoutfile = net_ + "/holdout_";
				  this->write_acts(testDataLoader.X(), holdoutfile);
			  }

		  }

		}

	void predict(TensorContainer<cpu, 4, real_t> &xtest, std::vector<int> &ytest){
		// mini-batch per device
		  int batch_size = 100;
		  int num_out = nets_[0]->get_outputdim();
		  int step = batch_size / ndev_;

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

			for (index_t j = 0; j + batch_size <= xtest.size(0); j += batch_size) {
			  nets_[tid]->Forward(xtest.Slice(j + tid * step, j + (tid + 1) * step), pred, false);
			  for (int k = 0; k < step; ++ k) {
				nerr   += MaxIndex(pred[k]) != ytest[j + tid * step + k];
				logloss += (save_log(pred[ k ][ytest[j + tid * step + k]]));
			  }
			}
		  }

		  printf("%.2f%% ", (1.0 - (real_t)nerr/xtest.size(0))*100);
		  printf("logloss %.4f ", (-(real_t)logloss/xtest.size(0)));
	}

	void predict_batch(TensorContainer<cpu, 4, real_t> &xtest, std::vector<int> &ytest, long & ext_nerr, long & ext_logloss){
		// mini-batch per device
		  int batch_size = 100;
		  int num_out = nets_[0]->get_outputdim();
		  int step = batch_size / ndev_;

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

			for (index_t j = 0; j + batch_size <= xtest.size(0); j += batch_size) {
			  nets_[tid]->Forward(xtest.Slice(j + tid * step, j + (tid + 1) * step), pred, false);
			  for (int k = 0; k < step; ++ k) {
				nerr   += MaxIndex(pred[k]) != ytest[j + tid * step + k];
				logloss += (save_log(pred[ k ][ytest[j + tid * step + k]]));
			  }
			}
		  }

		  ext_nerr += nerr;
		  ext_logloss += logloss;
	}

	void write_acts(TensorContainer<cpu, 4, real_t> &xtest, std::string outputfile){
			// mini-batch per device
		  int batch_size = 100;
		  int step = batch_size / ndev_;
		  int num_out = nets_[0]->get_outputdim();

		  // evaluation
		  mshadow::SetDevice<xpu>(devs_[0]);
		  TensorContainer<cpu, 2, real_t> pred;
		  pred.Resize(Shape2(step, num_out));

		  for (index_t j = 0; j + step <= xtest.size(0); j += step) {
				nets_[0]->Forward(xtest.Slice(j, j + step), pred, false);
				//Save activations
				nets_[0]->save_activations(22, outputfile+"softmax.csv");
		  }
	}


	~nntrainer(){
	 for(int i = 0; i < ndev_; ++i) {
		  mshadow::SetDevice<xpu>(devs_[i]);
		  delete nets_[i];
		  ShutdownTensorEngine<xpu>();
	  }
	  delete(myIA_);
	}

private:

	ImageAugmenter* myIA_;
	mshadow::ps::ISharedModel<xpu, real_t> *ps_;
	int ndev_;
	std::string net_;
	std::vector<int> devs_;
	std::vector<INNet *> nets_;
	TensorContainer<cpu, 4, real_t> xtrain_augmented_;
    std::string logfile_;

};


#endif /* NNTRAINER_H_ */
