#include "iter_base.h"
#include "iter_imglst.h"
#include "iter_threadbuffer.h"

#include "../utility/util.h"

namespace iter{

IIter< DataChunk > * CreateIter(std::string net, const configVec & conf, bool isTrain) {
  IIter< DataChunk > * InputIter = NULL;
  for ( unsigned int i = 0; i < conf.size(); ++i ) {
    
    const char* name = conf[i].first.c_str();
    const char* val = conf[i].second.c_str();
   
    if ( !strcmp(name, "iter") ) {
      if ( !strcmp(val, "imglst" ) ) {
	InputIter = new ImglstIter(net, isTrain);
	continue;
      }
      if ( !strcmp(val, "threadbuffer" ) ) {
	if ( InputIter == NULL ) {
	  utility::Error("Must specify Iterator Type before Buffer Type");
	  exit(-1);
	}
	InputIter = new ThreadBufferIter(InputIter);
	continue;
      }
      else {
	utility::Error("Unknown Iterator Type: %s", val);
      }
    }
  }
  
  if ( InputIter != NULL ) {
      InputIter->SetParam(conf);
  }
  else {
    utility::Error("No data Iterator specified");
    exit(-1);
  }

  return InputIter;
}
 
}

