/*
 * mshadow_op.h
 *
 *  Created on: Aug 23, 2015
 *      Author: niklas
 */

#ifndef MSHADOW_OP_H_
#define MSHADOW_OP_H_

#include <mshadow/tensor.h>

// define operations
struct relu_op{
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return max(a, 0.0f);
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};
struct leaky_relu_op{
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return a > 0.0f ? a : (a / 3.0f);
  }
};
struct leaky_relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : (1.0f/3.0f);
  }
};
struct threshold {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a < b ? 1.0f : 0.0f;
  }
};
struct clip {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    if (isnan(a)) return 0.0f;
    if (a < -b) return -b;
    if (a > b) return b;
    return a;
  }
};
struct clean_nan {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    if (isnan(a)) return 0.0f;
    return a;
  }
};
struct addconst {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return (a + b);
  }
};

real_t save_log(real_t val){
	if(val <= 0.00000000001){
		return logf(0.00000000001);
	}else if(val >= 0.99999999999){
		return logf(0.99999999999);
	}
	else{
		return logf(val);
	}
}



#endif /* MSHADOW_OP_H_ */
