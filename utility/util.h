#pragma once
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstdarg>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

typedef float real_t;

namespace utility {

using namespace mshadow;
using namespace std;


// helper function to messure wall time
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
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
void write_val_to_file(const char* outputfile, T val, bool newline){

	  std::ofstream outputstream (outputfile, ios::app);

	  if (outputstream.is_open()){
		  if(newline){
			  outputstream << val << std::endl;
		  }else{
			  outputstream << val << ",";
		  }
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

inline bool file_existent (const std::string& name) {
	  struct stat buffer;
	  return (stat (name.c_str(), &buffer) == 0);
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
