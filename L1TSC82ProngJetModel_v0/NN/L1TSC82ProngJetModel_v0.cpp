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
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer2_t>(layer2_out, "batch_normalization", N_INPUT_1_1*N_INPUT_2_1);
#endif

    phi1_result_t layer29_out[N_OUTPUTS_29*N_FILT_29];
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, phi1_result_t, config29>(layer2_out, layer29_out, w29, b29); // phi1
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<phi1_result_t>(layer29_out, "phi1", N_OUTPUTS_29*N_FILT_29);
#endif

    layer5_t layer5_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::hard_tanh<phi1_result_t, layer5_t, hard_tanh_config5>(layer29_out, layer5_out); // q_activation
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer5_t>(layer5_out, "q_activation", N_LAYER_1_3*N_LAYER_2_3);
#endif

    phi2_result_t layer30_out[N_OUTPUTS_30*N_FILT_30];
    #pragma HLS ARRAY_PARTITION variable=layer30_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer5_t, phi2_result_t, config30>(layer5_out, layer30_out, w30, b30); // phi2
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<phi2_result_t>(layer30_out, "phi2", N_OUTPUTS_30*N_FILT_30);
#endif

    layer8_t layer8_out[N_LAYER_1_6*N_LAYER_2_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::hard_tanh<phi2_result_t, layer8_t, hard_tanh_config8>(layer30_out, layer8_out); // q_activation_1
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer8_t>(layer8_out, "q_activation_1", N_LAYER_1_6*N_LAYER_2_6);
#endif

    phi3_result_t layer31_out[N_OUTPUTS_31*N_FILT_31];
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer8_t, phi3_result_t, config31>(layer8_out, layer31_out, w31, b31); // phi3
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<phi3_result_t>(layer31_out, "phi3", N_OUTPUTS_31*N_FILT_31);
#endif

    layer11_t layer11_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::hard_tanh<phi3_result_t, layer11_t, hard_tanh_config11>(layer31_out, layer11_out); // q_activation_2
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer11_t>(layer11_out, "q_activation_2", N_LAYER_1_9*N_LAYER_2_9);
#endif

    layer12_t layer12_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::linear<layer11_t, layer12_t, linear_config12>(layer11_out, layer12_out); // q_activation_3
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer12_t>(layer12_out, "q_activation_3", N_LAYER_1_9*N_LAYER_2_9);
#endif

    layer13_t layer13_out[N_FILT_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::global_pooling1d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // global_max_pooling1d
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer13_t>(layer13_out, "global_max_pooling1d", N_FILT_13);
#endif

    rho1_result_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, rho1_result_t, config14>(layer13_out, layer14_out, w14, b14); // rho1
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<rho1_result_t>(layer14_out, "rho1", N_LAYER_14);
#endif

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::hard_tanh<rho1_result_t, layer16_t, hard_tanh_config16>(layer14_out, layer16_out); // q_activation_4
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer16_t>(layer16_out, "q_activation_4", N_LAYER_14);
#endif

    rho2_result_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, rho2_result_t, config17>(layer16_out, layer17_out, w17, b17); // rho2
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<rho2_result_t>(layer17_out, "rho2", N_LAYER_17);
#endif

    layer19_t layer19_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::hard_tanh<rho2_result_t, layer19_t, hard_tanh_config19>(layer17_out, layer19_out); // q_activation_5
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer19_t>(layer19_out, "q_activation_5", N_LAYER_17);
#endif

    rho3_result_t layer20_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::dense<layer19_t, rho3_result_t, config20>(layer19_out, layer20_out, w20, b20); // rho3
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<rho3_result_t>(layer20_out, "rho3", N_LAYER_20);
#endif

    layer22_t layer22_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::hard_tanh<rho3_result_t, layer22_t, hard_tanh_config22>(layer20_out, layer22_out); // q_activation_6
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer22_t>(layer22_out, "q_activation_6", N_LAYER_20);
#endif

    layer23_t layer23_out[N_LAYER_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // output
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<layer23_t>(layer23_out, "output", N_LAYER_23);
#endif

    nnet::sigmoid<layer23_t, result_t, sigmoid_config25>(layer23_out, layer25_out); // output_sigmoid_activation
#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
    nnet::save_layer_output<result_t>(layer25_out, "output_sigmoid_activation", N_LAYER_23);
#endif

}

} // namespace hls4ml_L1TSC82ProngJetModel_v0