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

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<batch_normalization_scale_t, 17>(s2, "s2.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 17>(b2, "b2.txt");
        nnet::load_weights_from_txt<phi1_weight_t, 1088>(w18, "w18.txt");
        nnet::load_weights_from_txt<phi1_bias_t, 64>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight8_t, 2048>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight11_t, 128>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 4>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight14_t, 8>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 2>(b14, "b14.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(input_layer, layer2_out, s2, b2); // batch_normalization

    phi1_result_t layer18_out[N_OUTPUTS_18*N_FILT_18];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, phi1_result_t, config18>(layer2_out, layer18_out, w18, b18); // phi1

    layer5_t layer5_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<phi1_result_t, layer5_t, relu_config5>(layer18_out, layer5_out); // q_activation

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