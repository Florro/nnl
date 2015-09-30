/*
 * thread_util.h
 *
 *  Created on: Sep 27, 2015
 *      Author: florian
 */

#ifndef THREAD_UTIL_H_
#define THREAD_UTIL_H_

#include <pthread.h>

class Thread {
  
public:
  inline void Start(void * entry(void*), void* param);
  
  inline int Join(void);

  static inline void Exit(void* status);
  
private:
  pthread_t thread;
};

inline void Thread::Start(void * entry(void*), void* param) {
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&thread, &attr, entry, param);
}

inline int Thread::Join(void) {
  void *status;
  return pthread_join(thread, &status);
}

inline void Thread::Exit(void* status) {
  pthread_exit(status);
}

#endif /* THREAD_UTIL_H_ */