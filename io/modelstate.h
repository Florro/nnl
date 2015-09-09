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
struct CreateWeightSaverVisitor : public IVisitor<xpu> {

	int layer_id_;

	  // constructor
	  CreateWeightSaverVisitor(int layer_id) : layer_id_(layer_id) {}


	virtual void Visit( bool is_bias,
						mshadow::Tensor<xpu,1> weight,
	                    mshadow::Tensor<xpu,1> grad) {

		  //Write params into file
		  std::string outputfile = "/home/niklas/CXX/nnl/testNets/plankton/net1/modelstate/layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
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
		  std::string outputfile = "/home/niklas/CXX/nnl/testNets/plankton/net1/modelstate/layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
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
		  std::string outputfile = "/home/niklas/CXX/nnl/testNets/plankton/net1/modelstate/layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
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
		std::string outputfile = "/home/niklas/CXX/nnl/testNets/plankton/net1/modelstate/layer_" + utility::custom_to_string(layer_id_) + "_" + utility::custom_to_string(is_bias);
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
void save_weights( ILayer<xpu> *p_layer, int layer_id ){

	  CreateWeightSaverVisitor<xpu> visitor(layer_id);
	  p_layer->ApplyVisitor(&visitor);
}

}

#endif /* MODELSTATE_H_ */
