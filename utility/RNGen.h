/*
 * RNGen.h
 *
 *  Created on: Feb 3, 2015
 *      Author: niklas
 */

#ifndef RNGen_H_
#define RNGen_H_

#include <boost/thread/mutex.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

using namespace std;

class RNGen {
  timespec ts;
public:
  boost::mt19937 gen;
  boost::mutex RNGenseedGeneratorMutex;
  boost::mt19937 RNGenseedGenerator;

  RNGen() {
    clock_gettime(CLOCK_REALTIME, &ts);
    RNGenseedGeneratorMutex.lock();
    gen.seed(RNGenseedGenerator()+ts.tv_nsec);
    RNGenseedGeneratorMutex.unlock();
  }
  int randint(int n) {
    if (n==0) return 0;
    else return gen()%n;
  }
  float uniform(float a=0, float b=1) {
    unsigned int k=gen();
    return a+(b-a)*k/4294967296.0;
  }
  float normal(float mean=0, float sd=1) {
    boost::normal_distribution<> nd(mean, sd);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(gen, nd);
    return mean+sd*var_nor();
  }
  int bernoulli(float p) {
    if (uniform()<p)
      return 1;
    else
      return 0;
  }

};


#endif /* RNGen_H_ */
