#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"

//hls-fpga-machine-learning insert numbers
#define N_EDGE 37
#define LAYER7_OUT_DIM 1
#define TWO 2
#define NODE_DIM 3
#define LAYER6_OUT_DIM 3
#define LAYER5_OUT_DIM 4
#define EDGE_DIM 4
#define LAYER4_OUT_DIM 4
#define N_NODE 28

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,8> model_default_t;
typedef ap_fixed<16,8> input_t;
typedef ap_fixed<16,8> input2_t;
typedef ap_uint<16> input3_t;
typedef ap_fixed<16,8> layer4_t;
typedef ap_fixed<16,8> layer5_t;
typedef ap_fixed<16,8> layer6_t;
typedef ap_fixed<16,8> layer7_t;
typedef ap_fixed<16,8> result_t;

#endif
