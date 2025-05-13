#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w18.h"
#include "weights/b18.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w14.h"
#include "weights/b14.h"


// hls-fpga-machine-learning insert layer-config
// batch_normalization
struct config2 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1*N_INPUT_2_1;
    static const unsigned n_filt = 17;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_bias_t bias_t;
    typedef batch_normalization_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// phi1
struct config18_mult : nnet::dense_config {
    static const unsigned n_in = 136;
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef phi1_bias_t bias_t;
    typedef phi1_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config18 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 8;
    static const unsigned n_chan = 17;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 8;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 8;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 8;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_18<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef phi1_bias_t bias_t;
    typedef phi1_weight_t weight_t;
    typedef config18_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::pointwise_conv_18<data_T, res_T, CONFIG_T>;
};
const ap_uint<config18::filt_width> config18::pixels[] = {0};

// q_activation
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_activation_table_t table_t;
};

// q_activation_1
struct linear_config6 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_activation_1_table_t table_t;
};

// global_average_pooling1d
struct config7 : nnet::pooling1d_config {
    static const unsigned n_in = 8;
    static const unsigned n_filt = 64;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const unsigned reuse_factor = 1;
    typedef model_default_t accum_t;
};

// rho1
struct config8 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 32;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 28;
    static const unsigned n_nonzeros = 2020;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef layer8_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_activation_2
struct relu_config10 : nnet::activ_config {
    static const unsigned n_in = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_activation_2_table_t table_t;
};

// rho2
struct config11 : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 3;
    static const unsigned n_nonzeros = 125;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef layer11_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_activation_3
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_activation_3_table_t table_t;
};

// output
struct config14 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 8;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef layer14_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// output_linear
struct linear_config15 : nnet::activ_config {
    static const unsigned n_in = 2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef output_linear_table_t table_t;
};

// softmax
struct Softmax_config16 : nnet::activ_config {
    static const unsigned n_in = 2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef softmax_exp_table_t exp_table_t;
    typedef softmax_inv_table_t inv_table_t;
};



#endif
