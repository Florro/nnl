/*
 * semaphore_util.h
 *
 *  Created on: Sep 27, 2015
 *      Author: florian
 */

#ifndef SEMAPHORE_UTIL_H_
#define SEMAPHORE_UTIL_H_

#include <semaphore.h>

class Semaphore {
  
public:
  inline void Init(int initVal);
  
  inline void Destroy(void);
  
  inline void Wait(void);
  
  inline void Post(void);
  
private:
  sem_t sem;
};

inline void Semaphore::Init(int initVal) {
  sem_init(&sem, 0, initVal);
}

inline void Semaphore::Destroy(void) {
  sem_destroy(&sem);
}

inline void Semaphore::Wait(void) {
  sem_wait(&sem);
}

inline void Semaphore::Post(void) {
  sem_post(&sem);
}


#endif /* SEMAPHORE_UTIL_H_ */