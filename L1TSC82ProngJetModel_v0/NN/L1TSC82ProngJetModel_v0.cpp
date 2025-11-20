#include <iostream>

#include "L1TSC82ProngJetModel_v0.h"
#include "parameters.h"

namespace hls4ml_L1TSC82ProngJetModel_v0 {
void L1TSC82ProngJetModel_v0(
    input_t input_layer[8*20],
    result_t layer25_out[1]
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
        nnet::load_weights_from_txt<phi1_weight_t, 1040>(w26, "w26.txt");
        nnet::load_weights_from_txt<phi1_bias_t, 52>(b26, "b26.txt");
        nnet::load_weights_from_txt<phi2_weight_t, 1040>(w27, "w27.txt");
        nnet::load_weights_from_txt<phi2_bias_t, 20>(b27, "b27.txt");
        nnet::load_weights_from_txt<phi3_weight_t, 80>(w28, "w28.txt");
        nnet::load_weights_from_txt<phi3_bias_t, 4>(b28, "b28.txt");
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

    layer2_t layer2_out[8*20];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    phi1_result_t layer26_out[8*52];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0

    layer5_t layer5_out[8*52];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    phi2_result_t layer27_out[8*20];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0

    layer8_t layer8_out[8*20];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    phi3_result_t layer28_out[8*4];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0

    layer11_t layer11_out[8*4];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    layer12_t layer12_out[8*4];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    layer13_t layer13_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    rho1_result_t layer14_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    layer16_t layer16_out[36];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    rho2_result_t layer17_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0

    layer19_t layer19_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0

    rho3_result_t layer20_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0

    layer22_t layer22_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0

    layer23_t layer23_out[1];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0

    nnet::normalize<input_t, layer2_t, config2>(input_layer, layer2_out, s2, b2); // batch_normalization
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer2_t>(layer2_out, "batch_normalization", 8*20);
#endif

    nnet::pointwise_conv_1d_cl<layer2_t, phi1_result_t, config29>(layer2_out, layer26_out, w26, b26); // phi1
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<phi1_result_t>(layer26_out, "phi1", 8*52);
#endif

    nnet::hard_tanh<phi1_result_t, layer5_t, hard_tanh_config5>(layer26_out, layer5_out); // q_activation
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer5_t>(layer5_out, "q_activation", 8*52);
#endif

    nnet::pointwise_conv_1d_cl<layer5_t, phi2_result_t, config30>(layer5_out, layer27_out, w27, b27); // phi2
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<phi2_result_t>(layer27_out, "phi2", 8*20);
#endif

    nnet::hard_tanh<phi2_result_t, layer8_t, hard_tanh_config8>(layer27_out, layer8_out); // q_activation_1
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer8_t>(layer8_out, "q_activation_1", 8*20);
#endif

    nnet::pointwise_conv_1d_cl<layer8_t, phi3_result_t, config31>(layer8_out, layer28_out, w28, b28); // phi3
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<phi3_result_t>(layer28_out, "phi3", 8*4);
#endif

    nnet::hard_tanh<phi3_result_t, layer11_t, hard_tanh_config11>(layer28_out, layer11_out); // q_activation_2
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer11_t>(layer11_out, "q_activation_2", 8*4);
#endif

    nnet::linear<layer11_t, layer12_t, linear_config12>(layer11_out, layer12_out); // q_activation_3
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer12_t>(layer12_out, "q_activation_3", 8*4);
#endif

    nnet::global_pooling1d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // global_max_pooling1d
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer13_t>(layer13_out, "global_max_pooling1d", 4);
#endif

    nnet::dense<layer13_t, rho1_result_t, config14>(layer13_out, layer14_out, w14, b14); // rho1
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<rho1_result_t>(layer14_out, "rho1", 36);
#endif

    nnet::hard_tanh<rho1_result_t, layer16_t, hard_tanh_config16>(layer14_out, layer16_out); // q_activation_4
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer16_t>(layer16_out, "q_activation_4", 36);
#endif

    nnet::dense<layer16_t, rho2_result_t, config17>(layer16_out, layer17_out, w17, b17); // rho2
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<rho2_result_t>(layer17_out, "rho2", 4);
#endif

    nnet::hard_tanh<rho2_result_t, layer19_t, hard_tanh_config19>(layer17_out, layer19_out); // q_activation_5
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer19_t>(layer19_out, "q_activation_5", 4);
#endif

    nnet::dense<layer19_t, rho3_result_t, config20>(layer19_out, layer20_out, w20, b20); // rho3
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<rho3_result_t>(layer20_out, "rho3", 4);
#endif

    nnet::hard_tanh<rho3_result_t, layer22_t, hard_tanh_config22>(layer20_out, layer22_out); // q_activation_6
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer22_t>(layer22_out, "q_activation_6", 4);
#endif

    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // output
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer23_t>(layer23_out, "output", 1);
#endif

    nnet::sigmoid<layer23_t, result_t, sigmoid_config25>(layer23_out, layer25_out); // output_sigmoid_activation
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<result_t>(layer25_out, "output_sigmoid_activation", 1);
#endif

}
} //namespace hls4ml_L1TSC82ProngJetModel_v0 
