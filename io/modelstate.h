/*
 * modelstate.h
 *
 *  Created on: Sep 9, 2015
 *      Author: niklas
 */

#ifndef MODELSTATE_H_
#define MODELSTATE_H_

#include "../layer/layer.h"

namespace modelstate{

//Creates a visitor, which creates correct updaters for each individual weight structure
template<typename xpu>
struct CreateWeightLoaderVisitor : public IVisitor<xpu> {

	int layer_id_;
	std::string inputfile_;

	  // constructor
	CreateWeightLoaderVisitor(int layer_id, std::string inputfile) : layer_id_(layer_id), inputfile_(inputfile) {}


	virtual void Visit( bool is_bias,
						mshadow::Tensor<xpu,1> weight,
	                    mshadow::Tensor<xpu,1> grad) {

		  TensorContainer<cpu, 1, real_t> host_weight;
		  host_weight.Resize(weight.shape_);

		  //Write params into file
		  std::string inputfile = inputfile_ + "layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);

		  std::ifstream dataSet ((char*)inputfile.c_str());
		  if(!dataSet){
			  utility::Error("Loading weights at layer %i from file %s failed", layer_id_, (char*)inputfile.c_str());
		  }
		  unsigned count = 0;
		  while (dataSet)
		  {
			  std::string s;
			  if (!std::getline( dataSet, s )) break;
			  host_weight[count] = atof(s.c_str());
			  count++;
		  }
		  dataSet.close();

		  Copy(weight, host_weight, weight.stream_);

	}
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,2> weight,
					   mshadow::Tensor<xpu,2> grad) {

		  TensorContainer<cpu, 2, real_t> host_weight;
		  host_weight.Resize(weight.shape_);

		  //Write params into file
		  std::string inputfile = inputfile_ + "layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);

		  std::ifstream dataSet ((char*)inputfile.c_str());
		  if(!dataSet){
			  utility::Error("Loading weights at layer %i from file %s failed", layer_id_, (char*)inputfile.c_str());
		  }
		  unsigned count1 = 0;
		  unsigned count2 = 0;
		  while (dataSet)
		  {
			  std::string s;
			  if (!std::getline( dataSet, s )) break;
			  count1++;
			  std::istringstream ss( s );
			  count2 = 0;
			  while (ss)
			  {
				std::string s;
				if (!getline( ss, s, ',' )) break;
				host_weight[count1][count2] = atof(s.c_str());
				count2++;
			  }
		  }
		  dataSet.close();

		  Copy(weight, host_weight, weight.stream_);

  	}
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,3> weight,
					   mshadow::Tensor<xpu,3> grad) {

  	}
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,4> weight,
					   mshadow::Tensor<xpu,4> grad) {
	}

};

//Creates a visitor, which creates correct updaters for each individual weight structure
template<typename xpu>
struct CreateWeightSaverVisitor : public IVisitor<xpu> {

	int layer_id_;
	std::string outputfile_;

	  // constructor
	  CreateWeightSaverVisitor(int layer_id, std::string outputfile) : layer_id_(layer_id), outputfile_(outputfile) {}


	virtual void Visit( bool is_bias,
						mshadow::Tensor<xpu,1> weight,
	                    mshadow::Tensor<xpu,1> grad) {

		  //Write params into file
		  std::string outputfile = outputfile_ + "layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
		  std::ofstream outputstream ((char*)outputfile.c_str());


		  TensorContainer<cpu, 1, real_t> host_weight;
		  host_weight.Resize(weight.shape_);
		  Copy(host_weight, weight, weight.stream_);


		  if (outputstream.is_open()){
			  for(int i = 0; i < host_weight.size(0); i++){
				  outputstream << host_weight[i] << std::endl;
			  }
		  outputstream.close();
		  }
		  else{
			  utility::Error("Saving weights at layer %i failed", layer_id_);
		  }

	}
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,2> weight,
					   mshadow::Tensor<xpu,2> grad) {

		  TensorContainer<cpu, 2, real_t> host_weight;
		  host_weight.Resize(weight.shape_);
		  Copy(host_weight, weight, weight.stream_);

		  //Write params into file
		  std::string outputfile = outputfile_ + "layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
		  std::ofstream outputstream ((char*)outputfile.c_str());
		  if (outputstream.is_open()){
			  for(int i = 0; i < host_weight.size(0); i++){
				  for(int j = 0; j < host_weight.size(1); j++){
					  outputstream << host_weight[i][j] << ",";
				  }
				  outputstream << std::endl;
			  }
		  outputstream.close();
		  }
		  else{
			  utility::Error("Saving weights at layer %i failed", layer_id_);
		  }

  	}
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,3> weight,
					   mshadow::Tensor<xpu,3> grad) {
		 //Write params into file
		  std::string outputfile = outputfile_ + "layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
		  std::ofstream outputstream ((char*)outputfile.c_str());

		  TensorContainer<cpu, 3, real_t> host_weight;
		  host_weight.Resize(weight.shape_);
		  Copy(host_weight, weight, weight.stream_);

		  if (outputstream.is_open()){
			  for(int i = 0; i < host_weight.size(0); i++){
				  for(int j = 0; j < host_weight.size(1); j++){
					  for(int k = 0; k < host_weight.size(2); k++){
						  outputstream << host_weight[i][j][k] << ",";
					  }
					  outputstream << std::endl;
				  }
				  outputstream << std::endl << std::endl;
			  }
		  outputstream.close();
		  }
		  else{
			  utility::Error("Saving weights at layer %i failed", layer_id_);
		  }


  	}
	virtual void Visit(bool is_bias,
					   mshadow::Tensor<xpu,4> weight,
					   mshadow::Tensor<xpu,4> grad) {

		//Write params into file
		std::string outputfile = outputfile_ + "layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
		std::ofstream outputstream ((char*)outputfile.c_str());

		TensorContainer<cpu, 4, real_t> host_weight;
		host_weight.Resize(weight.shape_);
		Copy(host_weight, weight, weight.stream_);

		if (outputstream.is_open()){
			for(int i = 0; i < host_weight.size(0); i++){
			  for(int j = 0; j < host_weight.size(1); j++){
				  for(int k = 0; k < host_weight.size(2); k++){
					  for(int p = 0; p < host_weight.size(3); p++){
						  outputstream << host_weight[i][j][k][p] << ",";
					  }
					  outputstream << std::endl;
				  }
				  outputstream << std::endl << std::endl;
			  }
			  outputstream << std::endl << std::endl;
		 }
		outputstream.close();
		}
		else{
		  utility::Error("Saving weights at layer %i failed", layer_id_);
		}

  	}

};

//Creates Asyncupdaters
template<typename xpu>
void save_weights( ILayer<xpu> *p_layer, int layer_id, std::string outputfile ){

	  CreateWeightSaverVisitor<xpu> visitor(layer_id, outputfile);
	  p_layer->ApplyVisitor(&visitor);
}

//Creates Asyncupdaters
template<typename xpu>
void load_weights( ILayer<xpu> *p_layer, int layer_id, std::string outputfile ){

	  CreateWeightLoaderVisitor<xpu> visitor(layer_id, outputfile);
	  p_layer->ApplyVisitor(&visitor);
}


}

#endif /* MODELSTATE_H_ */
