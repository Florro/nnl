/* ToDo
 *
 * 1)  initialization for deep nets
 * 2)  make datalayer construction optional in configurator / generate dim in code and write in config
 * 3)  trainvalidate just one input, and percentage test to train
 * 4)  config for dynamic augmentation
 * 5)  load params
 * 6)  multithread augmentation?
 * 7)  dynamic datastream
 * 8)  background image solution for different colors
 * 9)  in batchsizechanged activations seem not to be freed, just allocated again
 * 10) try 2D conv kernel
 * 11) find solution for image means (store in binary if not existent generate bin)
 * 12) Save modelstate (hyperparams, epoch etc.)
 * 14) 1/batchsize hardcode in sgdupdater
 * 15) Check cudnn pooling
 * 16) use average pooling in pool layer
 *
 *
 */


#include <time.h>
#include <sys/time.h>
#include "neuralnet/nntrainer.h"
#include "mshadow/tensor.h"
#include "neuralnet/configurator.h"



// multithreaded run routine
template<typename xpu>
inline int Run(int argc, char *argv[]) {

  std::string train_path;
  std::string test_path;

  //Read config file
  std::string net(argv[1]);
  std::vector < std::pair <std::string, std::string > > cfg = configurator::readcfg(net + "/config.conf");
  std::string mode = configurator::getmode(cfg);

  //Create nn trainer
  nntrainer<xpu>* mynntrainer = new nntrainer<xpu>(net, cfg);

  //train routine
  double wall0 = utility::get_wall_time();
  if(mode == "train"){
	  mynntrainer->trainvalidate_batchwise( );
  }else if (mode == "predict"){
	  mynntrainer->predict( );
  }
  else{
	  utility::Error("Unknown control parameter: %s, use train/predict", argv[argc-1]);
  }


  double wall1 = utility::get_wall_time();

  std::cout << "\nWall Time = " << wall1 - wall0 << std::endl;

  return 0;
}



int main(int argc, char *argv[]) {

  if (argc < 2) {
    printf("Usage: nnl <config>");
    return 0;
  }
  Run<mshadow::gpu>(argc, argv);

  return 0;

}














