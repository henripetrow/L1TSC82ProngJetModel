#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> layer2_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> batch_normalization_scale_t;
typedef ap_fixed<24,12,AP_RND,AP_SAT,0> batch_normalization_bias_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<38,19> phi1_result_t;
typedef ap_fixed<8,1> phi1_weight_t;
typedef ap_fixed<8,1> phi1_bias_t;
typedef ap_ufixed<2,0> q_activation_slope_prec;
typedef ap_ufixed<2,0> q_activation_shift_prec;
typedef ap_fixed<8,1,AP_RND,AP_SAT,0> layer5_t;
typedef ap_ufixed<2,0> slope5_t;
typedef ap_ufixed<2,0> shift5_t;
typedef ap_fixed<18,8> q_activation_table_t;
typedef ap_fixed<23,9> phi2_result_t;
typedef ap_fixed<8,1> phi2_weight_t;
typedef ap_fixed<8,1> phi2_bias_t;
typedef ap_ufixed<2,0> q_activation_1_slope_prec;
typedef ap_ufixed<2,0> q_activation_1_shift_prec;
typedef ap_fixed<8,1,AP_RND,AP_SAT,0> layer8_t;
typedef ap_ufixed<2,0> slope8_t;
typedef ap_ufixed<2,0> shift8_t;
typedef ap_fixed<18,8> q_activation_1_table_t;
typedef ap_fixed<22,8> phi3_result_t;
typedef ap_fixed<8,1> phi3_weight_t;
typedef ap_fixed<8,1> phi3_bias_t;
typedef ap_ufixed<2,0> q_activation_2_slope_prec;
typedef ap_ufixed<2,0> q_activation_2_shift_prec;
typedef ap_fixed<8,1,AP_RND,AP_SAT,0> layer11_t;
typedef ap_ufixed<2,0> slope11_t;
typedef ap_ufixed<2,0> shift11_t;
typedef ap_fixed<18,8> q_activation_2_table_t;
typedef ap_fixed<20,11,AP_RND,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> q_activation_3_table_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<27,10> rho1_result_t;
typedef ap_fixed<8,1> weight14_t;
typedef ap_fixed<8,1> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<2,0> q_activation_4_slope_prec;
typedef ap_ufixed<2,0> q_activation_4_shift_prec;
typedef ap_fixed<8,1,AP_RND,AP_SAT,0> layer16_t;
typedef ap_ufixed<2,0> slope16_t;
typedef ap_ufixed<2,0> shift16_t;
typedef ap_fixed<18,8> q_activation_4_table_t;
typedef ap_fixed<23,9> rho2_result_t;
typedef ap_fixed<8,1> weight17_t;
typedef ap_fixed<8,1> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_ufixed<2,0> q_activation_5_slope_prec;
typedef ap_ufixed<2,0> q_activation_5_shift_prec;
typedef ap_fixed<8,1,AP_RND,AP_SAT,0> layer19_t;
typedef ap_ufixed<2,0> slope19_t;
typedef ap_ufixed<2,0> shift19_t;
typedef ap_fixed<18,8> q_activation_5_table_t;
typedef ap_fixed<19,5> rho3_result_t;
typedef ap_fixed<8,1> weight20_t;
typedef ap_fixed<8,1> bias20_t;
typedef ap_uint<1> layer20_index;
typedef ap_ufixed<2,0> q_activation_6_slope_prec;
typedef ap_ufixed<2,0> q_activation_6_shift_prec;
typedef ap_fixed<8,1,AP_RND,AP_SAT,0> layer22_t;
typedef ap_ufixed<2,0> slope22_t;
typedef ap_ufixed<2,0> shift22_t;
typedef ap_fixed<18,8> q_activation_6_table_t;
typedef ap_fixed<20,10,AP_RND,AP_SAT,0> layer23_t;
typedef ap_fixed<9,3> weight23_t;
typedef ap_fixed<9,3> bias23_t;
typedef ap_uint<1> layer23_index;
typedef ap_ufixed<20,10,AP_RND,AP_SAT,0> result_t;
typedef ap_fixed<18,8> output_sigmoid_activation_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
