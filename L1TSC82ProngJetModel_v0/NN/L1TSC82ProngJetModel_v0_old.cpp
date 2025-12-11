#include <iostream>

#include "L1TSC82ProngJetModel_v0.h"
#include "parameters.h"

namespace hls4ml_L1TSC82ProngJetModel_v0 {
void L1TSC82ProngJetModel_v0(
    input_t input_layer[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer25_out[N_LAYER_23]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_layer complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_layer,layer25_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<batch_normalization_scale_t, 20>(s2, "s2.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 20>(b2, "b2.txt");
        nnet::load_weights_from_txt<phi1_weight_t, 1040>(w29, "w29.txt");
        nnet::load_weights_from_txt<phi1_bias_t, 52>(b29, "b29.txt");
        nnet::load_weights_from_txt<phi2_weight_t, 1040>(w30, "w30.txt");
        nnet::load_weights_from_txt<phi2_bias_t, 20>(b30, "b30.txt");
        nnet::load_weights_from_txt<phi3_weight_t, 80>(w31, "w31.txt");
        nnet::load_weights_from_txt<phi3_bias_t, 4>(b31, "b31.txt");
        nnet::load_weights_from_txt<weight14_t, 144>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 36>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight17_t, 144>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 4>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight20_t, 16>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 4>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight23_t, 4>(w23, "w23.txt");
        nnet::load_weights_from_txt<bias23_t, 1>(b23, "b23.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(input_layer, layer2_out, s2, b2); // batch_normalization

    phi1_result_t layer29_out[N_OUTPUTS_29*N_FILT_29];
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, phi1_result_t, config29>(layer2_out, layer29_out, w29, b29); // phi1

    layer5_t layer5_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::hard_tanh<phi1_result_t, layer5_t, hard_tanh_config5>(layer29_out, layer5_out); // q_activation

    phi2_result_t layer30_out[N_OUTPUTS_30*N_FILT_30];
    #pragma HLS ARRAY_PARTITION variable=layer30_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer5_t, phi2_result_t, config30>(layer5_out, layer30_out, w30, b30); // phi2

    layer8_t layer8_out[N_LAYER_1_6*N_LAYER_2_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::hard_tanh<phi2_result_t, layer8_t, hard_tanh_config8>(layer30_out, layer8_out); // q_activation_1

    phi3_result_t layer31_out[N_OUTPUTS_31*N_FILT_31];
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer8_t, phi3_result_t, config31>(layer8_out, layer31_out, w31, b31); // phi3

    layer11_t layer11_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::hard_tanh<phi3_result_t, layer11_t, hard_tanh_config11>(layer31_out, layer11_out); // q_activation_2

    layer12_t layer12_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::linear<layer11_t, layer12_t, linear_config12>(layer11_out, layer12_out); // q_activation_3

    layer13_t layer13_out[N_FILT_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::global_pooling1d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // global_max_pooling1d

    layer14_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // rho1

    layer15_t layer15_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::linear<layer14_t, layer15_t, linear_config15>(layer14_out, layer15_out); // rho1_linear

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::hard_tanh<layer15_t, layer16_t, hard_tanh_config16>(layer15_out, layer16_out); // q_activation_4

    rho2_result_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, rho2_result_t, config17>(layer16_out, layer17_out, w17, b17); // rho2

    layer19_t layer19_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::hard_tanh<rho2_result_t, layer19_t, hard_tanh_config19>(layer17_out, layer19_out); // q_activation_5

    rho3_result_t layer20_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::dense<layer19_t, rho3_result_t, config20>(layer19_out, layer20_out, w20, b20); // rho3

    layer22_t layer22_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::hard_tanh<rho3_result_t, layer22_t, hard_tanh_config22>(layer20_out, layer22_out); // q_activation_6

    layer23_t layer23_out[N_LAYER_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // output

    nnet::sigmoid<layer23_t, result_t, sigmoid_config25>(layer23_out, layer25_out); // output_sigmoid_activation




std::cout << "'input' : [" ;
for (int i = 0; i < 8*20; i++) {
    std::cout << (double)input_layer[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "'batch_n' : [" ;
for (int i = 0; i < 8*20; i++) {
    std::cout << (double)layer2_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "'phi1' : [" ;
for (int i = 0; i < 8*52; i++) {
    std::cout << (double)layer29_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "'phi2' : [" ;
for (int i = 0; i < 8*20; i++) {
    std::cout << layer30_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "'phi3' : [" ;
for (int i = 0; i < 8*4; i++) {
    std::cout << layer31_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "'global_max_pooling1d' : [" ;
for (int i = 0; i < 4; i++) {
    std::cout << layer13_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "'rho1' : [" ;
for (int i = 0; i < 36; i++) {
    std::cout << layer14_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "rho2[" ;
for (int i = 0; i < 4; i++) {
    std::cout << layer17_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "rho3[" ;
for (int i = 0; i < 4; i++) {
    std::cout << layer20_out[i] << ",";
}
std::cout << "]," << std::endl;

std::cout << "output[" ;
for (int i = 0; i < 1; i++) {
    std::cout << layer23_out[i];
}
std::cout << "]," << std::endl;

std::cout << "output_act[" ;
for (int i = 0; i < 1; i++) {
    std::cout << layer25_out[i];
}
std::cout << "]" << std::endl;





}

} // namespace hls4ml_L1TSC82ProngJetModel_v0