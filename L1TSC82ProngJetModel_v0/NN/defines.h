#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 8
#define N_INPUT_2_1 17
#define N_INPUT_1_1 8
#define N_INPUT_2_1 17
#define N_OUTPUTS_17 8
#define N_FILT_17 64
#define N_LAYER_1_3 8
#define N_LAYER_2_3 64
#define N_LAYER_1_3 8
#define N_LAYER_2_3 64
#define N_FILT_7 64
#define N_LAYER_8 32
#define N_LAYER_8 32
#define N_LAYER_11 4
#define N_LAYER_11 4
#define N_LAYER_14 2
#define N_LAYER_14 2
#define N_LAYER_14 2


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer2_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> batch_normalization_scale_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> batch_normalization_bias_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<38,19> phi1_result_t;
typedef ap_fixed<8,1> phi1_weight_t;
typedef ap_fixed<8,1> phi1_bias_t;
typedef ap_ufixed<8,0,AP_RND,AP_SAT,0> layer5_t;
typedef ap_fixed<18,8> q_activation_table_t;
typedef ap_fixed<20,11,AP_RND,AP_SAT,0> layer6_t;
typedef ap_fixed<18,8> q_activation_1_table_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<31,14> rho1_result_t;
typedef ap_fixed<8,1> weight8_t;
typedef ap_fixed<8,1> bias8_t;
typedef ap_uint<1> layer8_index;
typedef ap_ufixed<8,0,AP_RND,AP_SAT,0> layer10_t;
typedef ap_fixed<18,8> q_activation_2_table_t;
typedef ap_fixed<22,7> rho2_result_t;
typedef ap_fixed<8,1> weight11_t;
typedef ap_fixed<8,1> bias11_t;
typedef ap_uint<1> layer11_index;
typedef ap_ufixed<8,0,AP_RND,AP_SAT,0> layer13_t;
typedef ap_fixed<18,8> q_activation_3_table_t;
typedef ap_ufixed<24,12,AP_RND,AP_SAT,0> layer14_t;
typedef ap_fixed<9,3> weight14_t;
typedef ap_fixed<9,3> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<18,8> output_linear_table_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> softmax_inv_table_t;


#endif
