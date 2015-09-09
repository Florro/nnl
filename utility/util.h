#pragma once
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "mshadow/tensor.h"

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


void load_data_list(const char* filename, std::vector < std::pair < int, std::string > > &img_location){
      std::ifstream dataSet (filename, std::ios::in);
      assert(dataSet);
      while (dataSet)
      {
        std::string s;
        if (!std::getline( dataSet, s )) break;
        std::pair < int, std::string > tmp;
        std::istringstream ss( s );
        int count = 0;
        while (ss)
        {
          std::string s;
          if (!getline( ss, s, ',' )) break;
          else if ( count % 2 == 0 ) tmp.first = atoi(s.c_str());
		  else if ( count % 2 == 1 ) tmp.second = s;
          count++;
        }
        img_location.push_back(tmp);
      }
      dataSet.close();
}

void LoadImages( TensorContainer<cpu, 4, real_t> &xdata, vector<string> &img_locations, const unsigned int & start, const unsigned int & size, const int & nchannels){

		for (unsigned i = start; i < size; i++){

			cv::Mat img = cv::imread( (char*)img_locations[i].c_str(), cv::IMREAD_COLOR );

			/*
			cv::resize(img, img, cv::Size(512,512));
			//save img to file
			std::string img_loc = "/home/niklas/Desktop/retina_acts/textures/retina/" + custom_to_string(i) + ".jpg";
			cv::imwrite( img_loc, img);

			if(i == 1500){
				exit(0);
			}
			*/


			if(false){
				cv::namedWindow( "pic" );
				cv::resize(img, img, cv::Size(512,512));
				cv::imshow( "pic", img );
				cv::waitKey(0);
			}
			for(unsigned y = 0; y < xdata.size(2); ++y) {
			  for(unsigned x = 0; x < xdata.size(3); ++x) {
				cv::Vec3b bgr = img.at< cv::Vec3b >(y, x);
				// store in RGB order
				xdata[i][0][y][x] = bgr[0];
				if(nchannels == 3){
					xdata[i][1][y][x] = bgr[1];
					xdata[i][2][y][x] = bgr[2];
				}
			  }
			}

		}

}

void Load_Images_Labels( TensorContainer<cpu, 4, real_t> & xdata, std::vector< int > & ydata, std::vector < std::pair < int, std::string > > imglst, const unsigned int & start, const unsigned int & size, const int & nchannels){

		for (unsigned i = 0; i < size; i++){

			//load label
			ydata.push_back(imglst[start + i].first);

			//load image
			cv::Mat img = cv::imread( imglst[start + i].second, cv::IMREAD_COLOR );

			if(false){
				cv::namedWindow( "pic" );
				cv::resize(img, img, cv::Size(512,512));
				cv::imshow( "pic", img );
				cv::waitKey(0);
			}
			for(unsigned y = 0; y < xdata.size(2); ++y) {
			  for(unsigned x = 0; x < xdata.size(3); ++x) {
				cv::Vec3b bgr = img.at< cv::Vec3b >(y, x);
				// store in RGB order
				xdata[i][0][y][x] = bgr[0];
				if(nchannels == 3){
					xdata[i][1][y][x] = bgr[1];
					xdata[i][2][y][x] = bgr[2];
				}
			  }
			}

		}

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
