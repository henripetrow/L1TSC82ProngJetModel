#include <iostream>

#include "L1TSC82ProngJetModel_v0.h"
#include "parameters.h"

namespace hls4ml_L1TSC82ProngJetModel_v0 {
void L1TSC82ProngJetModel_v0(
    input_t input_layer[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer16_out[N_LAYER_14]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_layer complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_layer,layer16_out 
    #pragma HLS DATAFLOW


    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(input_layer, layer2_out, s2, b2); // batch_normalization

    phi1_result_t layer17_out[N_OUTPUTS_17*N_FILT_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, phi1_result_t, config18>(layer2_out, layer17_out, w17, b17); // phi1

    layer5_t layer5_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<phi1_result_t, layer5_t, relu_config5>(layer17_out, layer5_out); // q_activation

    layer6_t layer6_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::linear<layer5_t, layer6_t, linear_config6>(layer5_out, layer6_out); // q_activation_1

    layer7_t layer7_out[N_FILT_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::global_pooling1d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out); // global_average_pooling1d

    rho1_result_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, rho1_result_t, config8>(layer7_out, layer8_out, w8, b8); // rho1

    layer10_t layer10_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::relu<rho1_result_t, layer10_t, relu_config10>(layer8_out, layer10_out); // q_activation_2

    rho2_result_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer10_t, rho2_result_t, config11>(layer10_out, layer11_out, w11, b11); // rho2

    layer13_t layer13_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<rho2_result_t, layer13_t, relu_config13>(layer11_out, layer13_out); // q_activation_3

    layer14_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // output

    layer15_t layer15_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::linear<layer14_t, layer15_t, linear_config15>(layer14_out, layer15_out); // output_linear

    nnet::softmax<layer15_t, result_t, Softmax_config16>(layer15_out, layer16_out); // softmax

}

} // namespace hls4ml_L1TSC82ProngJetModel_v0
