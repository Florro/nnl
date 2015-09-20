/*
 * configurator.h
 *
 *  Created on: Aug 20, 2015
 *      Author: niklas
 */

#ifndef CONFIGURATOR_H_
#define CONFIGURATOR_H_

#include <iostream>
#include <string>
#include <fstream>
#include <memory>


#include "../layer/layer.h"
#include "../layer/data_layer.h"
#include "../layer/standard_layer.h"
#include "../layer/dropout_layer.h"
#include "../layer/activations_layer.h"
#include "../layer/flatten_layer.h"
#include "../layer/pool_layer.h"
#include "../layer/pool_cudnn_layer.h"
#include "../layer/conv_layer.h"
#include "../layer/conv_cudnn_layer.h"
#include "../layer/softmax_layer.h"
#include "../layer/batchnorm_layer.h"

#include "mshadow/tensor.h"
#include "../utility/mshadow_op.h"
#include "../utility/image_augmenter.h"

namespace configurator{

using namespace std;


/*!
 * \brief base implementation of config reader
 */
class ConfigReaderBase {
 public:
  /*!
   * \brief get current name, called after Next returns true
   * \return current parameter name
   */
  inline const char *name(void) const {
    return s_name.c_str();
  }
  /*!
   * \brief get current value, called after Next returns true
   * \return current parameter value
   */
  inline const char *val(void) const {
    return s_val.c_str();
  }
  /*!
   * \brief move iterator to next position
   * \return true if there is value in next position
   */
  inline bool Next(void) {
    while (!this->IsEnd()) {
      GetNextToken(&s_name);
      if (s_name == "=") return false;
      if (GetNextToken(&s_buf) || s_buf != "=")  return false;
      if (GetNextToken(&s_val) || s_val == "=")  return false;
      return true;
    }
    return false;
  }
  // called before usage
  inline void Init(void) {
    ch_buf = this->GetChar();
  }

 protected:
  /*!
   * \brief to be implemented by subclass,
   * get next token, return EOF if end of file
   */
  virtual char GetChar(void) = 0;

  /*! \brief to be implemented by child, utility::Check if end of stream */
  virtual bool IsEnd(void) = 0;

 private:
  char ch_buf;
  std::string s_name, s_val, s_buf;

  inline void SkipLine(void) {
    do {
      ch_buf = this->GetChar();
    } while (ch_buf != EOF && ch_buf != '\n' && ch_buf != '\r');
  }

  // return newline
  inline bool GetNextToken(std::string *tok) {
    tok->clear();
    bool new_line = false;
    while (ch_buf != EOF) {
      switch (ch_buf) {
        case '#' : SkipLine(); new_line = true; break;
        case '=':
          if (tok->length() == 0) {
            ch_buf = this->GetChar();
            *tok = '=';
          }
          return new_line;
        case '[' :
        case ']' :
        case '\r':
        case '\n':
          if (tok->length() == 0) new_line = true;
        case '\t':
        case ' ' :
          ch_buf = this->GetChar();
          if (tok->length() != 0) return new_line;
          break;
        default:
          *tok += ch_buf;
          ch_buf = this->GetChar();
          break;
      }
    }
    if (tok->length() == 0) {
      return true;
    } else {
      return false;
    }
  }
};
/*!
 * \brief an iterator use stream base, allows use all types of istream
 */
class ConfigStreamReader: public ConfigReaderBase {
 public:
  /*!
   * \brief constructor
   * \param istream input stream
   */
  explicit ConfigStreamReader(std::istream &fin) : fin(fin) {}

 protected:
  virtual char GetChar(void) {
    return fin.get();
  }
  /*! \brief to be implemented by child, utility::Check if end of stream */
  virtual bool IsEnd(void) {
    return fin.eof();
  }

 private:
  std::istream &fin;
};
/*!
 * \brief an iterator that iterates over a configure file and gets the configures
 */
class ConfigIterator: public ConfigStreamReader {
 public:
  /*!
   * \brief constructor
   * \param fname name of configure file
   */
  ConfigIterator(const char *fname) : ConfigStreamReader(fi) {
    fi.open(fname);
    if (fi.fail()) {
      utility::Error("cannot open file %s", fname);
    }
    ConfigReaderBase::Init();
  }

  bool isEnd(void){
	 return ConfigStreamReader::IsEnd();
  }

  /*! \brief destructor */
  ~ConfigIterator(void) {
    fi.close();
  }

 private:
  std::ifstream fi;
};


template<typename xpu>
void setSoftmaxLayer(std::vector< ILayer<xpu>* > &architecture, Loss_layer<xpu>*& losslayer){
	losslayer = new Softmax_layer<xpu>(architecture.back());
}

bool getpair(const char* &name, const char* &val, std::pair<std::string, std::string> line){

	name = line.first.c_str();
	val  = line.second.c_str();
	if(!strcmp(name, "Layer")){
		return false;
	}else if(!strcmp(name, "Loss")){
		return false;
	}
	else if(!strcmp(name, "GlobalParams")){
		return false;
	}
	else if(!strcmp(name, "Dataconfig")){
		return false;
	}

	return true;
}

template<typename xpu>
void setDataLayer(std::vector < std::pair <std::string, std::string > > &cfg, int &i, std::vector< ILayer<xpu>* > &architecture, mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

	int dim_z = 1;
	int dim_y = 1;
	int dim_x = 1;

	const char *name = "empty";
	const char *val = "empty";

	i++;
	while(getpair(name, val, cfg[i])){
		if(!strcmp(name, "dim_x")){
			dim_x = atoi(val);
			utility::Check(dim_x > 0, "Datalayer dimension x has to be greater than 0");
		}
		else if(!strcmp(name, "dim_y")){
			dim_y = atoi(val);
			utility::Check(dim_y > 0, "Datalayer dimension y has to be greater than 0");
		}
		else if(!strcmp(name, "dim_z")){
			dim_z = atoi(val);
			utility::Check(dim_z > 0, "Datalayer dimension z has to be greater than 0");
		}
		else{
			utility::Error("Unknown Datalayer Layer Parameter: %s", name);
		}
		i++;
	}
	architecture.push_back(new Data_layer<xpu>(dim_z, dim_y, dim_x));
	architecture.back()->InitLayer(stream,rnd);
	i--;
}

template<typename xpu>
void setConvolutionLayer(std::vector < std::pair <std::string, std::string > > &cfg, int &i, std::vector< ILayer<xpu>* > &architecture, mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

	int filter_sl = 3;
	int filter_n = 32;
	int stride = 1;
	int padding = 0;
	real_t weightsInit = 0.0f;

	const char *name = "empty";
	const char *val = "empty";

	i++;
	while(getpair(name, val, cfg[i])){
		if(!strcmp(name, "filter_sl")){
			filter_sl = atoi(val);
			utility::Check(filter_sl > 0, "conv Filter size has to be positive");
		}
		else if(!strcmp(name, "filter_n")){
			filter_n = atoi(val);
			utility::Check(filter_n > 1, "conv Filter number has to be greater than 1");
		}
		else if(!strcmp(name, "stride")){
			if(!strcmp(val, "off")){
				stride = 1;
			}else{
				stride = atoi(val);
			}
			utility::Check(stride > 0, "conv stride has to be at least 1");
		}
		else if(!strcmp(name, "padding")){
			if(!strcmp(val, "off")){
				padding = 0;
			}else{
				padding = atoi(val);
			}
			utility::Check(padding >= 0, "conv padding has to be positive");
		}
		else if(!strcmp(name, "weightsInit")){
			weightsInit = atof(val);
			utility::Check(weightsInit >= 0, "weightInit has to be positive");
		}
		else{
			utility::Error("Unknown Convolution Layer Parameter: %s", name);
		}
		i++;
	}

	architecture.push_back(new Conv_cudnn_layer<xpu>(architecture.back(), filter_sl, stride, filter_n, padding, weightsInit ));
	architecture.back()->InitLayer(stream,rnd);
	i--;
}

template<typename xpu>
void setActivationLayer(std::vector < std::pair <std::string, std::string > > &cfg, int &i, std::vector< ILayer<xpu>* > &architecture, mshadow::Stream<xpu> *stream,  Random<xpu, real_t> &rnd){

	const char *name = cfg[i].first.c_str();
    const char *val = cfg[i].second.c_str();

	i++;
	while(getpair(name, val, cfg[i])){

    	if(!strcmp(name, "act_type")){
    		if(!strcmp(val, "ReLU")){
    			architecture.push_back(new activation_layer<xpu, relu_op, relu_grad>(architecture.back()));
    			architecture.back()->InitLayer(stream,rnd);
    		}
    		else if(!strcmp(val, "leakyReLU")){
    			architecture.push_back(new activation_layer<xpu, leaky_relu_op, leaky_relu_grad>(architecture.back()));
				architecture.back()->InitLayer(stream,rnd);
    		}
    		else{
				utility::Error("Unknown Activation Type: %s", val);
			}
		}
    	else{
    		utility::Error("Unknown Activation Layer Parameter: %s", name);
    	}
    	i++;
    }
	i--;

}

template<typename xpu>
void setStandardLayer(std::vector < std::pair <std::string, std::string > > &cfg, int &i, std::vector< ILayer<xpu>* > &architecture, mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

	int layerDim = 100;

	const char *name = "empty";
	const char *val = "empty";
	i++;
	while(getpair(name, val, cfg[i])){

		if(!strcmp(name, "layerDim")){
			layerDim = atoi(val);
			utility::Check(layerDim > 1, "standard layerdim has to be greater than 1");
		}
		else{
			utility::Error("Unknown Standard Layer Parameter: %s", name);
		}
		i++;
	}

	if(!(architecture.back()->getpAct()->is_mat())){
		architecture.push_back(new flatten_layer<xpu>(architecture.back()));
		architecture.back()->InitLayer(stream,rnd);
	}
	architecture.push_back(new Standard_layer<xpu>(architecture.back(), layerDim));
	architecture.back()->InitLayer(stream,rnd);
	i--;
}

template<typename xpu>
void setDropoutLayer(std::vector < std::pair <std::string, std::string > > &cfg, int &i, std::vector< ILayer<xpu>* > &architecture, mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

	real_t dropout = 0.5f;

	const char *name = "empty";
	const char *val = "empty";

	i++;
	while(getpair(name, val, cfg[i])){
		if(!strcmp(name, "dropout")){
			dropout = atof(val);
			utility::Check(dropout < 1.0f && dropout > 0.0f, "dropout must be between 0 and 1");
		}
		else{
			utility::Error("Unknown Dropout Layer Parameter: %s", name);
		}
		i++;
	}

	architecture.push_back(new Dropout_layer<xpu>(architecture.back(), dropout));
	architecture.back()->InitLayer(stream,rnd);
	i--;
}

template<typename xpu>
void setBatchNormLayer(std::vector < std::pair <std::string, std::string > > &cfg, int &i, std::vector< ILayer<xpu>* > &architecture, mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

    real_t init_slope = 1.0f;
    real_t init_bias = 0.0f;
    real_t eps = 1e-10f;
    real_t bn_momentum = 0.9f;


	const char *name = "empty";
	const char *val = "empty";

	i++;
	while(getpair(name, val, cfg[i])){
		if(!strcmp(name, "init_slope")){
			init_slope = atof(val);
			utility::Check(init_slope > 0, "batch norm init slope must be positiv");
		}
		else if(!strcmp(name, "bias")){
			init_bias = atof(val);
		}
		else if(!strcmp(name, "eps")){
			eps = atof(val);
			utility::Check(eps > 0, "batch norm epsilon must be positiv");
		}
		else if(!strcmp(name, "momentum")){
			bn_momentum = atof(val);
			utility::Check(bn_momentum > 0 && bn_momentum < 1, "batch norm momentum must be between 0 and 1");
		}
		else{
			utility::Error("Unknown Dropout Layer Parameter: %s", name);
		}
		i++;
	}

	architecture.push_back(new batchnorm_layer<xpu, false>(architecture.back(), init_slope,  init_bias,  eps,  bn_momentum));
	architecture.back()->InitLayer(stream,rnd);
	i--;
}

template<typename xpu>
void setPoolingLayer(std::vector < std::pair <std::string, std::string > > &cfg, int &i, std::vector< ILayer<xpu>* > &architecture, mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

	int window_sl = 3;
	int stride = 1;
	bool modeflag = false;

	const char *name = "empty";
	const char *val = "empty";
	i++;
	while(getpair(name, val, cfg[i])){
		if(!strcmp(name, "window_sl")){
			window_sl = atoi(val);
			utility::Check(window_sl > 1, "Pooling window_sl must be greater than 1");
		}
		else if(!strcmp(name, "stride")){
			if(!strcmp(val, "off")){
				stride = 1;
			}else{
				stride = atoi(val);
			}
			utility::Check(stride > 0, "Pooling stride must be at least 1");
		}
		else if(!strcmp(name, "mode")){
			if(!strcmp(val, "max")){
				architecture.push_back(new pool_layer<xpu, red::maximum>(architecture.back(), window_sl, stride));
				architecture.back()->InitLayer(stream,rnd);
			}else if(!strcmp(val, "avg")){
				architecture.push_back(new pool_cudnn_layer<xpu, AVG>(architecture.back(), window_sl, stride));
				architecture.back()->InitLayer(stream,rnd);
			}else{
				utility::Error("Unknown Pooling Mode: %s", val);
			}
			modeflag = true;
		}
		else{
			utility::Error("Unknown Pooling Layer Parameter: %s", name);
		}
		i++;
	}
	if(!modeflag){
		utility::Error("Pooling Layer needs Reducer specification (max/avg)");
	}
	i--;
}

void setSGDGlobalParams(std::vector < std::pair <std::string, std::string > > &cfg, int &i, updater::UpdaterParam* hyperparam){

	const char *name = "empty";
	const char *val = "empty";

	i++;
	while(getpair(name, val, cfg[i])){
			if(!strcmp(name, "lr_schedule")){
				hyperparam->lr_schedule = atoi(val);
				utility::Check(hyperparam->lr_schedule == 1 || hyperparam->lr_schedule == 0, "Learning schedule has to be 1 or 0: %i", val);
			}
			else if(!strcmp(name, "momentum_schedule")){
				hyperparam->momentum_schedule = atoi(val);
				utility::Check(hyperparam->momentum_schedule == 1 || hyperparam->momentum_schedule == 0, "Momentum schedule has to be 1 or 0: %i", val);
			}
			else if(!strcmp(name, "base_lr_")){
				hyperparam->base_lr_ = atof(val);
				utility::Check(hyperparam->base_lr_ > 0, "Base learning rate must be greater than 0: %i", val);
			}
			else if(!strcmp(name, "lr_decay")){
				hyperparam->lr_decay = atof(val);
				utility::Check(hyperparam->lr_decay > 0 && hyperparam->lr_decay < 1, "learning rate decay must be between 0 and 1: %i", val);
			}
			else if(!strcmp(name, "lr_minimum")){
				hyperparam->lr_minimum = atof(val);
				utility::Check(hyperparam->lr_minimum > 0, "learning rate minimum must be greater than 0: %i", val);
			}
			else if(!strcmp(name, "base_momentum")){
				hyperparam->base_momentum_ = atof(val);
				utility::Check(hyperparam->base_momentum_ > 0 && hyperparam->base_momentum_ < 1, "base momentum must be between 0 and 1: %i", val);
			}
			else if(!strcmp(name, "final_momentum")){
				hyperparam->final_momentum_ = atof(val);
				utility::Check(hyperparam->final_momentum_ > 0 && hyperparam->final_momentum_ < 1, "final momentum must be between 0 and 1: %i", val);
			}
			else if(!strcmp(name, "saturation_epoch")){
				hyperparam->saturation_epoch_ = atoi(val);
				utility::Check(hyperparam->saturation_epoch_ >= 0, "saturation epoch must be positive: %i", val);
			}
			else if(!strcmp(name, "weightdecay")){
				hyperparam->weightdecay = atof(val);
				utility::Check( hyperparam->weightdecay   >= 0 && hyperparam->weightdecay < 1,  "weightdecay must be between 0 and 1: %i", val );
			}
			else if(!strcmp(name, "clipgradient")){
				hyperparam->clipgradient = atof(val);
				utility::Check( hyperparam->clipgradient  >= 0,  "weightdecay must be positive: %i", val );
			}
			else if(!strcmp(name, "epochs")){
				hyperparam->epochs = atoi(val);
				utility::Check( hyperparam->epochs  >= 0,  "number of epochs must be positive", val );
			}
			else if(!strcmp(name, "batchsize")){
				hyperparam->batchsize = atoi(val);
				utility::Check( hyperparam->batchsize  > 0,  "batchsize must be atleast 1", val );
			}
			else{
				utility::Error("Unknown Global parameter: %s", name);
			}
			i++;
		}
	i--;

}


template<typename xpu>
void readNetConfig(std::vector < std::pair <std::string, std::string > > &cfg, std::vector< ILayer<xpu>* > &architecture, Loss_layer<xpu>*& losslayer, updater::UpdaterParam* hyperparam,  mshadow::Stream<xpu> *stream, Random<xpu, real_t> &rnd){

	bool netconfigmode = false;

	for(int i = 0; i < cfg.size(); i++){
		const char *name = cfg[i].first.c_str();
	    const char *val = cfg[i].second.c_str();

		if(!strcmp(name, "Netconfig")){
			netconfigmode = !strcmp(val, "start");
		}
		else if(!strcmp(name, "GlobalParams")){
			netconfigmode = strcmp(val, "start");
		}
		else if(!strcmp(name, "Dataconfig")){
			while(strcmp(name, "Dataconfig")){
				i++;
				name = cfg[i].first.c_str();
				val = cfg[i].second.c_str();
			}
		}
		else if(netconfigmode){
			if(!strcmp(name, "Layer")){
				if(!strcmp(val, "Data")){
					setDataLayer<xpu>(cfg, i, architecture, stream, rnd);
				}
				else if(!strcmp(val, "Convolution")){
					setConvolutionLayer<xpu>(cfg, i, architecture, stream, rnd);
				}
				else if(!strcmp(val, "Activation")){
					setActivationLayer<xpu>(cfg, i, architecture, stream, rnd);
				}
				else if(!strcmp(val, "Standard")){
					setStandardLayer<xpu>(cfg, i, architecture, stream, rnd);
				}
				else if(!strcmp(val, "Dropout")){
					setDropoutLayer<xpu>(cfg, i, architecture, stream, rnd);
				}
				else if(!strcmp(val, "Pooling")){
					setPoolingLayer<xpu>(cfg, i, architecture, stream, rnd);
				}
				else if (!strcmp(val, "Batchnorm")){
					setBatchNormLayer<xpu>(cfg,i, architecture, stream, rnd);
				}
				else{
					utility::Error("Unknown Layer Type: ", val);
				}
			}else if(!strcmp(name, "Loss")){
				if(!strcmp(val, "Softmax")){
					setSoftmaxLayer<xpu>(architecture, losslayer);
				}
				else{
					utility::Error("Unknown Loss Type: %s", val);
				}
			}
		}
		else if(!netconfigmode){ //Global params
			if(!strcmp(name, "Optimizer")){
				if(!strcmp(val, "SGD")){
					setSGDGlobalParams(cfg, i, hyperparam);
				}
				else{
					utility::Error("Unknown Minimizer: %s", val);
				}
			}


		}
		else{
			utility::Error("Incorrect config mode");
		}
	}

}


void setDataParameter(	std::vector < std::pair <std::string, std::string > > &cfg,
						int &i,
						std::string& trainpath,
						std::string& testpath,
						cvimg::augparams &augparameter	){



	const char *name = "empty";
	const char *val = "empty";

	augparameter.mirror = false;
	augparameter.scaling = 0.0;
	augparameter.translation = 0;
	augparameter.rotation = 0; //other axis commented out
	augparameter.sheering = 0.0f;
	augparameter.background = "white";
	augparameter.classweights_saturation_epoch = 1;

	while(getpair(name, val, cfg[i])){
			if(!strcmp(name, "trainpath")){
				trainpath = val;
			}
			else if(!strcmp(name, "testpath")){
				testpath = val;
			}
			else if(!strcmp(name, "mirror")){
				if(!strcmp(val, "true")){
					augparameter.mirror = true;
				}else if(!strcmp(val, "false")){
					augparameter.mirror  = false;
				}
				else{
					utility::Error("Wrong mirror flag: %s", val);
				}
			}
			else if(!strcmp(name, "scaling")){
				augparameter.scaling = atof(val);
				utility::Check(augparameter.scaling >= 0, "scaling must be positive: %i", val);
			}
			else if(!strcmp(name, "translation")){
				augparameter.translation = atoi(val);
				utility::Check(augparameter.translation >= 0, "translation must be positive: %i", val);
			}
			else if(!strcmp(name, "rotation")){
				augparameter.rotation = atoi(val);
				utility::Check(augparameter.rotation >= 0, "rotation must be positive: %i", val);
			}
			else if(!strcmp(name, "sheering")){
				augparameter.sheering = atof(val);
				utility::Check(augparameter.sheering >= 0, "sheering must be positive: %i", val);
			}
			else if(!strcmp(name, "background")){
				augparameter.background = val;
				utility::Check(augparameter.background == "white" || augparameter.background == "black" || augparameter.background == "gray", "background must be either black, white or gray %s", val);
			}
			else if(!strcmp(name, "classweights_start")){
				std::string str(val);
				std::string delimiter = ",";

				size_t pos = 0;
				std::string token;
				while ((pos = str.find(delimiter)) != std::string::npos) {
				    token = str.substr(0, pos);
				    augparameter.weights_start.push_back(atoi(token.c_str()));
				    str.erase(0, pos + delimiter.length());
				}
				augparameter.weights_start.push_back(atoi(str.c_str()));
			}
			else if(!strcmp(name, "classweights_end")){
				std::string str(val);
				std::string delimiter = ",";

				size_t pos = 0;
				std::string token;
				while ((pos = str.find(delimiter)) != std::string::npos) {
					token = str.substr(0, pos);
					augparameter.weights_end.push_back(atoi(token.c_str()));
					str.erase(0, pos + delimiter.length());
				}
				augparameter.weights_end.push_back(atoi(str.c_str()));
			}
			else if(!strcmp(name, "classweights_saturation_epoch")){
				augparameter.classweights_saturation_epoch = atoi(val);
				utility::Check(augparameter.classweights_saturation_epoch > 0, "classweights_saturation_epoch must be positive %s", val);
			}

			else{
				utility::Error("Unknown Dataconfig parameter: %s", name);
			}
			i++;
		}
	i--;


}

void readDataConfig(std::vector < std::pair <std::string, std::string > > &cfg,
					std::string &trainpath,
					std::string &testpath,
					cvimg::augparams &augparameter){

	bool dataconfigmode = false;

	for(int i = 0; i < (int)cfg.size(); i++){
		const char *name = cfg[i].first.c_str();
	    const char *val = cfg[i].second.c_str();

		if(!strcmp(name, "Netconfig")){
			dataconfigmode = false;
		}
		else if(!strcmp(name, "GlobalParams")){
			dataconfigmode = false;
		}
		else if(!strcmp(name, "Dataconfig")){
			dataconfigmode = !strcmp(val, "start");
		}
		else if(dataconfigmode){
			setDataParameter(cfg, i, trainpath, testpath, augparameter);
		}

	}

}



//read cfg
std::vector < std::pair <std::string, std::string > > readcfg(string cfgfile){
	std::vector < std::pair <std::string, std::string > > cfg;

	auto_ptr<ConfigIterator> myreader(new ConfigIterator((cfgfile).c_str()));

	while(!myreader->isEnd()){
		myreader->Next();
		cfg.push_back(std::make_pair(myreader->name(), myreader->val()));
	}


	return cfg;

}

//Get batchsize from config
int getbatchsize(std::vector< std::pair < std::string, std::string > > cfg){
	int batchsize = 0;

	for(unsigned i = 0; i < cfg.size(); i++){
		const char *name = cfg[i].first.c_str();
		const char *val = cfg[i].second.c_str();

		if(!strcmp(name, "batchsize")){
			batchsize = atoi(val);
		}
	}

	return batchsize;
}

//Get batchsize from config
int getepochs(std::vector< std::pair < std::string, std::string > > cfg){
	int epochs = 0;

	for(unsigned i = 0; i < cfg.size(); i++){
		const char *name = cfg[i].first.c_str();
		const char *val = cfg[i].second.c_str();

		if(!strcmp(name, "epochs")){
			epochs = atoi(val);
		}
	}

	return epochs;
}



}


#endif /* CONFIGURATOR_H_ */
