#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "nnet_conv1d_latency.h"
#include "nnet_helpers.h"

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_function_stubs.h"
#include "nnet_mult.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T> class PointwiseConv1D {
  public:
    static void pointwise_conv(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                               res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        // To be implemented in subclasses
    }
};

// hls4ml insert code
template<class data_T, typename CONFIG_T>
class fill_buffer_18 : public nnet::FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9]; buffer[0][10] =   data[10]; buffer[0][11] =   data[11]; buffer[0][12] =   data[12]; buffer[0][13] =   data[13]; buffer[0][14] =   data[14]; buffer[0][15] =   data[15]; buffer[0][16] =   data[16];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[17]; buffer[0][1] =   data[18]; buffer[0][2] =   data[19]; buffer[0][3] =   data[20]; buffer[0][4] =   data[21]; buffer[0][5] =   data[22]; buffer[0][6] =   data[23]; buffer[0][7] =   data[24]; buffer[0][8] =   data[25]; buffer[0][9] =   data[26]; buffer[0][10] =   data[27]; buffer[0][11] =   data[28]; buffer[0][12] =   data[29]; buffer[0][13] =   data[30]; buffer[0][14] =   data[31]; buffer[0][15] =   data[32]; buffer[0][16] =   data[33];

        }
        if (partition ==   2) {
            buffer[0][0] =   data[34]; buffer[0][1] =   data[35]; buffer[0][2] =   data[36]; buffer[0][3] =   data[37]; buffer[0][4] =   data[38]; buffer[0][5] =   data[39]; buffer[0][6] =   data[40]; buffer[0][7] =   data[41]; buffer[0][8] =   data[42]; buffer[0][9] =   data[43]; buffer[0][10] =   data[44]; buffer[0][11] =   data[45]; buffer[0][12] =   data[46]; buffer[0][13] =   data[47]; buffer[0][14] =   data[48]; buffer[0][15] =   data[49]; buffer[0][16] =   data[50];

        }
        if (partition ==   3) {
            buffer[0][0] =   data[51]; buffer[0][1] =   data[52]; buffer[0][2] =   data[53]; buffer[0][3] =   data[54]; buffer[0][4] =   data[55]; buffer[0][5] =   data[56]; buffer[0][6] =   data[57]; buffer[0][7] =   data[58]; buffer[0][8] =   data[59]; buffer[0][9] =   data[60]; buffer[0][10] =   data[61]; buffer[0][11] =   data[62]; buffer[0][12] =   data[63]; buffer[0][13] =   data[64]; buffer[0][14] =   data[65]; buffer[0][15] =   data[66]; buffer[0][16] =   data[67];

        }
        if (partition ==   4) {
            buffer[0][0] =   data[68]; buffer[0][1] =   data[69]; buffer[0][2] =   data[70]; buffer[0][3] =   data[71]; buffer[0][4] =   data[72]; buffer[0][5] =   data[73]; buffer[0][6] =   data[74]; buffer[0][7] =   data[75]; buffer[0][8] =   data[76]; buffer[0][9] =   data[77]; buffer[0][10] =   data[78]; buffer[0][11] =   data[79]; buffer[0][12] =   data[80]; buffer[0][13] =   data[81]; buffer[0][14] =   data[82]; buffer[0][15] =   data[83]; buffer[0][16] =   data[84];

        }
        if (partition ==   5) {
            buffer[0][0] =   data[85]; buffer[0][1] =   data[86]; buffer[0][2] =   data[87]; buffer[0][3] =   data[88]; buffer[0][4] =   data[89]; buffer[0][5] =   data[90]; buffer[0][6] =   data[91]; buffer[0][7] =   data[92]; buffer[0][8] =   data[93]; buffer[0][9] =   data[94]; buffer[0][10] =   data[95]; buffer[0][11] =   data[96]; buffer[0][12] =   data[97]; buffer[0][13] =   data[98]; buffer[0][14] =   data[99]; buffer[0][15] =  data[100]; buffer[0][16] =  data[101];

        }
        if (partition ==   6) {
            buffer[0][0] =  data[102]; buffer[0][1] =  data[103]; buffer[0][2] =  data[104]; buffer[0][3] =  data[105]; buffer[0][4] =  data[106]; buffer[0][5] =  data[107]; buffer[0][6] =  data[108]; buffer[0][7] =  data[109]; buffer[0][8] =  data[110]; buffer[0][9] =  data[111]; buffer[0][10] =  data[112]; buffer[0][11] =  data[113]; buffer[0][12] =  data[114]; buffer[0][13] =  data[115]; buffer[0][14] =  data[116]; buffer[0][15] =  data[117]; buffer[0][16] =  data[118];

        }
        if (partition ==   7) {
            buffer[0][0] =  data[119]; buffer[0][1] =  data[120]; buffer[0][2] =  data[121]; buffer[0][3] =  data[122]; buffer[0][4] =  data[123]; buffer[0][5] =  data[124]; buffer[0][6] =  data[125]; buffer[0][7] =  data[126]; buffer[0][8] =  data[127]; buffer[0][9] =  data[128]; buffer[0][10] =  data[129]; buffer[0][11] =  data[130]; buffer[0][12] =  data[131]; buffer[0][13] =  data[132]; buffer[0][14] =  data[133]; buffer[0][15] =  data[134]; buffer[0][16] =  data[135];

        }
    }
};
template<class data_T, class res_T, typename CONFIG_T>
class pointwise_conv_18 : public Conv1DKernel<data_T, res_T, CONFIG_T> {
  public:
    static void conv(
                     data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                     res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        data_T data_tmp[CONFIG_T::reuse_factor][CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor];
        #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
        res_T res_tmp[CONFIG_T::reuse_factor][CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor];
        #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0

    RFInputLoop:
        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
        #pragma HLS UNROLL
        InnerInputLoop:
            for (int ii = 0; ii < CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor; ii++) {
                #pragma HLS UNROLL
                data_tmp[jj][ii] = data[jj * CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor + ii];
            }
        }

        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);

    RFOutputLoop:
        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
        #pragma HLS UNROLL
        InnerOutputLoop:
            for (int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor; ii++) {
                #pragma HLS UNROLL
                res[jj * CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor + ii] = res_tmp[jj][ii];
            }
        }
    }
};

} // namespace nnet

#endif
