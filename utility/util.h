#pragma once
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

typedef float real_t;

namespace utility {

using namespace mshadow;
using namespace std;

int pack(unsigned char zz[4]){
    return (int)(zz[3]) 
        | (((int)(zz[2])) << 8)
        | (((int)(zz[1])) << 16)
        | (((int)(zz[0])) << 24);
}

template<typename T>
inline void shuffle(T *data, size_t sz){
  if(sz == 0) return;
  for(size_t i = sz - 1; i > 0; i--){
    std::swap(data[i], data[rand() % (i+1)]);
  } 
}
// random shuffle the data inside, require PRNG 
template<typename T>
inline void shuffle(std::vector<T> &data){
  shuffle(&data[0], data.size());
}

// simple function to load in mnist
inline void LoadMNIST(const char *path_img, const char *path_label,
                      std::vector<int> &ylabel,
                      TensorContainer<cpu, 2, real_t> &xdata,
                      bool do_shuffle){
  // load in data
  FILE *fi = fopen(path_img, "rb");
  if (fi == NULL) {
    printf("cannot open %s\n", path_img);
    exit(-1);
  }
  unsigned char zz[4];
  unsigned char *t_data, *l_data;
  int num_image, width, height, nlabel;            
  assert(fread(zz, 4 , 1, fi));
  assert(fread(zz, 4 , 1, fi));    
  num_image = pack(zz);
  assert(fread(zz, 4 , 1, fi));                
  width = pack(zz);
  assert(fread(zz, 4 , 1, fi));                    
  height = pack(zz);
  
  int step = width * height;
  t_data = new unsigned char[num_image * step];    
  assert(fread(t_data, step*num_image , 1 , fi));
  fclose(fi);
  
  // load in label
  fi = fopen(path_label, "rb");
  assert(fread(zz, 4 , 1, fi));
  assert(fread(zz, 4 , 1, fi));    
  nlabel = pack(zz);
  assert(num_image == nlabel);
  l_data = new unsigned char[num_image];
  assert(fread(l_data, num_image , 1 , fi));    
  // try to do shuffle 
  std::vector<int> rindex;
  for (int i = 0; i < num_image; ++ i) {
    rindex.push_back(i);
  }
  if (do_shuffle) {
    shuffle(rindex);
  }
  
  // save out result
  ylabel.resize(num_image);
  xdata.Resize(Shape2(num_image, width * height));
  for (int i = 0 ; i < num_image ; ++i) {
    for(int j = 0; j < step; ++j) {
      xdata[i][j] = (float)(t_data[rindex[i]*step + j]);
    }        
    ylabel[i] = l_data[rindex[i]];
  }
  delete[] t_data; delete [] l_data;
  printf("finish loading %dx%d matrix from %s, shuffle=%d\n", num_image, step, path_img, (int)do_shuffle);
}

const int kPrintBuffer = 1 << 12;

inline void HandleCheckError(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(-1);
}
inline void Check(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}
/*! \brief report error message, same as check */
inline void Error(const char *fmt, ...) {
  {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}


std::string custom_to_string( const int n ){
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}

template< typename T >
void write_vec_to_file(const char* outputfile, std::vector< T > &data){

	  std::ofstream outputstream (outputfile);

	  std::cout << "writing" << std::endl;
	  if (outputstream.is_open()){
	  for (int p = 0; p < data.size(); p++){
		outputstream << data[p] << std::endl;
	  }
	  outputstream.close();
	  }
	  else{
		  Error("Datafile not found %s", outputfile);
	  }

}

template< typename T >
void write_val_to_file(const char* outputfile, T val){

	  std::ofstream outputstream (outputfile, ios::app);

	  if (outputstream.is_open()){
		  outputstream << val << std::endl;
	  outputstream.close();
	  }
	  else{
		  Error("Datafile not found %s", outputfile);
	  }

}

void createDir(string NetDir, string name){
	struct stat st = {0};

	if (stat((char*)(NetDir + name).c_str(), &st) == -1) {		mkdir((char*)(NetDir + name).c_str(), 0700);	}
	else{			cout << "Directory " << name <<  " already exists!" << endl;	}
}



/*

void mean(TensorContainer<cpu, 4, real_t> &xtrain){

	TensorContainer<cpu, 4, real_t> xmeans;
	xmeans.Resize(Shape4(xtrain.size(0),3,1,1));
	xmeans = pool<red::sum>(xtrain, xmeans[0][0].shape_, xmeans.size(2), xmeans.size(3), 1);
	for(int i = 0; i < xmeans.size(0); i++){
		std::cout << xmeans[i][0][0][0] / xmeans.size(0) << std::endl;
	}

}
*/

}
