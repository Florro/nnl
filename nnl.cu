/* ToDo
 *
 * 3)  initialization for deep nets
 * 4)  make datalayer construction optional in configurator / generate dim in code and write in config
 * 6)  trainvalidate just one input, and percentage test to train
 * 8)  config for dynamic augmentation
 * 12) save params
 * 13) multithread augmentation?
 * 14) dynamic datastream
 * 17) img to bin
 * 18) background image solution for different colors
 *
 */

/* NOTES
 * cudnn pooling stochastic! -> pooling max, mshadow
 */

#include <time.h>
#include <sys/time.h>
#include "neuralnet/nntrainer.h"
#include "mshadow/tensor.h"


// helper function to messure wall time
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void read_data_mnist( TensorContainer<cpu, 4, real_t> &xtrain,  TensorContainer<cpu, 4, real_t> &xtest,
		std::vector<int> &ytrain, std::vector<int> &ytest){
	 // settings
	 int insize = 28;
	 srand(0);

	 // data
	 TensorContainer<cpu, 2, real_t> xtrain_, xtest_;
	 utility::LoadMNIST("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", ytrain, xtrain_, true);
	 utility::LoadMNIST("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", ytest, xtest_, false);

	 std::cout << std::endl;

	 xtrain.Resize(Shape4(xtrain_.size(0), 1, insize, insize));
	 xtest.Resize(Shape4(xtest_.size(0),  1, insize, insize));
	 xtrain = reshape(xtrain_, xtrain.shape_);
	 xtest = reshape(xtest_, xtest.shape_);
}

// multithreaded run routine
template<typename xpu>
inline int Run(int argc, char *argv[]) {


  TensorContainer<cpu, 4, real_t> xtrain;
  TensorContainer<cpu, 4, real_t> xtest;
  std::vector<int> ytrain, ytest;

  std::string train_path;
  std::string test_path;

  //choose data:
  int data = 1; //0 MNIST, 1 Plankton, 2 retina

  std::vector< std::string > imglst_train;

  //generate nn trainer
  std::string config;
  if(data == 0){
	  read_data_mnist(xtrain, xtest, ytrain, ytest);
	  config = "/home/niklas/CXX/conv_dist_NNL2/testNets/mnist.conf";
  }else if (data == 1){
	  config = "/home/niklas/CXX/conv_dist_NNL2/testNets/plankton.conf";
	  train_path = "/home/niklas/CXX/conv_dist_NNL2/data/plankton/trainnew.lst";
	  test_path = "/home/niklas/CXX/conv_dist_NNL2/data/plankton/testnew.lst";
  }else if (data == 2){
	  //imglst_train = read_data_retina_batch(xtest, ytrain, ytest);
	  //read_data_retina(xtrain, xtest, ytrain, ytest);
	  //config = "/home/niklas/CXX/conv_dist_NNL2/testNets/retina_learning_3class_test.conf";
  }
  nntrainer<xpu>* mynntrainer = new nntrainer<xpu>(argc, argv, config);

  //train routine
  double wall0 = get_wall_time();
  mynntrainer->trainvalidate_batchwise( train_path , test_path, false);
  double wall1 = get_wall_time();

  std::cout << "\nWall Time = " << wall1 - wall0 << std::endl;

  return 0;
}




int main(int argc, char *argv[]) {

  if (argc < 3) {
    printf("Usage: <device> devicelist\n"\
           "\tExample1: ./nnet_ps cpu 1 2 3\n"\
           "\tExample2: ./nnet_ps gpu 0 1\n");
    return 0;
  }
  if (!strcmp(argv[1], "cpu")) {
    Run<mshadow::cpu>(argc, argv);
  } else {
    Run<mshadow::gpu>(argc, argv);
  }


  return 0;

}
