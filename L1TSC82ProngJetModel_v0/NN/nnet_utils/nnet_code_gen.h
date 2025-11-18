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
class fill_buffer_29 : public nnet::FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9]; buffer[0][10] =   data[10]; buffer[0][11] =   data[11]; buffer[0][12] =   data[12]; buffer[0][13] =   data[13]; buffer[0][14] =   data[14]; buffer[0][15] =   data[15]; buffer[0][16] =   data[16]; buffer[0][17] =   data[17]; buffer[0][18] =   data[18]; buffer[0][19] =   data[19];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[20]; buffer[0][1] =   data[21]; buffer[0][2] =   data[22]; buffer[0][3] =   data[23]; buffer[0][4] =   data[24]; buffer[0][5] =   data[25]; buffer[0][6] =   data[26]; buffer[0][7] =   data[27]; buffer[0][8] =   data[28]; buffer[0][9] =   data[29]; buffer[0][10] =   data[30]; buffer[0][11] =   data[31]; buffer[0][12] =   data[32]; buffer[0][13] =   data[33]; buffer[0][14] =   data[34]; buffer[0][15] =   data[35]; buffer[0][16] =   data[36]; buffer[0][17] =   data[37]; buffer[0][18] =   data[38]; buffer[0][19] =   data[39];

        }
        if (partition ==   2) {
            buffer[0][0] =   data[40]; buffer[0][1] =   data[41]; buffer[0][2] =   data[42]; buffer[0][3] =   data[43]; buffer[0][4] =   data[44]; buffer[0][5] =   data[45]; buffer[0][6] =   data[46]; buffer[0][7] =   data[47]; buffer[0][8] =   data[48]; buffer[0][9] =   data[49]; buffer[0][10] =   data[50]; buffer[0][11] =   data[51]; buffer[0][12] =   data[52]; buffer[0][13] =   data[53]; buffer[0][14] =   data[54]; buffer[0][15] =   data[55]; buffer[0][16] =   data[56]; buffer[0][17] =   data[57]; buffer[0][18] =   data[58]; buffer[0][19] =   data[59];

        }
        if (partition ==   3) {
            buffer[0][0] =   data[60]; buffer[0][1] =   data[61]; buffer[0][2] =   data[62]; buffer[0][3] =   data[63]; buffer[0][4] =   data[64]; buffer[0][5] =   data[65]; buffer[0][6] =   data[66]; buffer[0][7] =   data[67]; buffer[0][8] =   data[68]; buffer[0][9] =   data[69]; buffer[0][10] =   data[70]; buffer[0][11] =   data[71]; buffer[0][12] =   data[72]; buffer[0][13] =   data[73]; buffer[0][14] =   data[74]; buffer[0][15] =   data[75]; buffer[0][16] =   data[76]; buffer[0][17] =   data[77]; buffer[0][18] =   data[78]; buffer[0][19] =   data[79];

        }
        if (partition ==   4) {
            buffer[0][0] =   data[80]; buffer[0][1] =   data[81]; buffer[0][2] =   data[82]; buffer[0][3] =   data[83]; buffer[0][4] =   data[84]; buffer[0][5] =   data[85]; buffer[0][6] =   data[86]; buffer[0][7] =   data[87]; buffer[0][8] =   data[88]; buffer[0][9] =   data[89]; buffer[0][10] =   data[90]; buffer[0][11] =   data[91]; buffer[0][12] =   data[92]; buffer[0][13] =   data[93]; buffer[0][14] =   data[94]; buffer[0][15] =   data[95]; buffer[0][16] =   data[96]; buffer[0][17] =   data[97]; buffer[0][18] =   data[98]; buffer[0][19] =   data[99];

        }
        if (partition ==   5) {
            buffer[0][0] =  data[100]; buffer[0][1] =  data[101]; buffer[0][2] =  data[102]; buffer[0][3] =  data[103]; buffer[0][4] =  data[104]; buffer[0][5] =  data[105]; buffer[0][6] =  data[106]; buffer[0][7] =  data[107]; buffer[0][8] =  data[108]; buffer[0][9] =  data[109]; buffer[0][10] =  data[110]; buffer[0][11] =  data[111]; buffer[0][12] =  data[112]; buffer[0][13] =  data[113]; buffer[0][14] =  data[114]; buffer[0][15] =  data[115]; buffer[0][16] =  data[116]; buffer[0][17] =  data[117]; buffer[0][18] =  data[118]; buffer[0][19] =  data[119];

        }
        if (partition ==   6) {
            buffer[0][0] =  data[120]; buffer[0][1] =  data[121]; buffer[0][2] =  data[122]; buffer[0][3] =  data[123]; buffer[0][4] =  data[124]; buffer[0][5] =  data[125]; buffer[0][6] =  data[126]; buffer[0][7] =  data[127]; buffer[0][8] =  data[128]; buffer[0][9] =  data[129]; buffer[0][10] =  data[130]; buffer[0][11] =  data[131]; buffer[0][12] =  data[132]; buffer[0][13] =  data[133]; buffer[0][14] =  data[134]; buffer[0][15] =  data[135]; buffer[0][16] =  data[136]; buffer[0][17] =  data[137]; buffer[0][18] =  data[138]; buffer[0][19] =  data[139];

        }
        if (partition ==   7) {
            buffer[0][0] =  data[140]; buffer[0][1] =  data[141]; buffer[0][2] =  data[142]; buffer[0][3] =  data[143]; buffer[0][4] =  data[144]; buffer[0][5] =  data[145]; buffer[0][6] =  data[146]; buffer[0][7] =  data[147]; buffer[0][8] =  data[148]; buffer[0][9] =  data[149]; buffer[0][10] =  data[150]; buffer[0][11] =  data[151]; buffer[0][12] =  data[152]; buffer[0][13] =  data[153]; buffer[0][14] =  data[154]; buffer[0][15] =  data[155]; buffer[0][16] =  data[156]; buffer[0][17] =  data[157]; buffer[0][18] =  data[158]; buffer[0][19] =  data[159];

        }
    }
};
template<class data_T, class res_T, typename CONFIG_T>
class pointwise_conv_29 : public Conv1DKernel<data_T, res_T, CONFIG_T> {
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
template<class data_T, typename CONFIG_T>
class fill_buffer_30 : public nnet::FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9]; buffer[0][10] =   data[10]; buffer[0][11] =   data[11]; buffer[0][12] =   data[12]; buffer[0][13] =   data[13]; buffer[0][14] =   data[14]; buffer[0][15] =   data[15]; buffer[0][16] =   data[16]; buffer[0][17] =   data[17]; buffer[0][18] =   data[18]; buffer[0][19] =   data[19]; buffer[0][20] =   data[20]; buffer[0][21] =   data[21]; buffer[0][22] =   data[22]; buffer[0][23] =   data[23]; buffer[0][24] =   data[24]; buffer[0][25] =   data[25]; buffer[0][26] =   data[26]; buffer[0][27] =   data[27]; buffer[0][28] =   data[28]; buffer[0][29] =   data[29]; buffer[0][30] =   data[30]; buffer[0][31] =   data[31]; buffer[0][32] =   data[32]; buffer[0][33] =   data[33]; buffer[0][34] =   data[34]; buffer[0][35] =   data[35]; buffer[0][36] =   data[36]; buffer[0][37] =   data[37]; buffer[0][38] =   data[38]; buffer[0][39] =   data[39]; buffer[0][40] =   data[40]; buffer[0][41] =   data[41]; buffer[0][42] =   data[42]; buffer[0][43] =   data[43]; buffer[0][44] =   data[44]; buffer[0][45] =   data[45]; buffer[0][46] =   data[46]; buffer[0][47] =   data[47]; buffer[0][48] =   data[48]; buffer[0][49] =   data[49]; buffer[0][50] =   data[50]; buffer[0][51] =   data[51]; buffer[0][52] =   data[52]; buffer[0][53] =   data[53]; buffer[0][54] =   data[54]; buffer[0][55] =   data[55]; buffer[0][56] =   data[56]; buffer[0][57] =   data[57]; buffer[0][58] =   data[58]; buffer[0][59] =   data[59]; buffer[0][60] =   data[60]; buffer[0][61] =   data[61]; buffer[0][62] =   data[62]; buffer[0][63] =   data[63];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[64]; buffer[0][1] =   data[65]; buffer[0][2] =   data[66]; buffer[0][3] =   data[67]; buffer[0][4] =   data[68]; buffer[0][5] =   data[69]; buffer[0][6] =   data[70]; buffer[0][7] =   data[71]; buffer[0][8] =   data[72]; buffer[0][9] =   data[73]; buffer[0][10] =   data[74]; buffer[0][11] =   data[75]; buffer[0][12] =   data[76]; buffer[0][13] =   data[77]; buffer[0][14] =   data[78]; buffer[0][15] =   data[79]; buffer[0][16] =   data[80]; buffer[0][17] =   data[81]; buffer[0][18] =   data[82]; buffer[0][19] =   data[83]; buffer[0][20] =   data[84]; buffer[0][21] =   data[85]; buffer[0][22] =   data[86]; buffer[0][23] =   data[87]; buffer[0][24] =   data[88]; buffer[0][25] =   data[89]; buffer[0][26] =   data[90]; buffer[0][27] =   data[91]; buffer[0][28] =   data[92]; buffer[0][29] =   data[93]; buffer[0][30] =   data[94]; buffer[0][31] =   data[95]; buffer[0][32] =   data[96]; buffer[0][33] =   data[97]; buffer[0][34] =   data[98]; buffer[0][35] =   data[99]; buffer[0][36] =  data[100]; buffer[0][37] =  data[101]; buffer[0][38] =  data[102]; buffer[0][39] =  data[103]; buffer[0][40] =  data[104]; buffer[0][41] =  data[105]; buffer[0][42] =  data[106]; buffer[0][43] =  data[107]; buffer[0][44] =  data[108]; buffer[0][45] =  data[109]; buffer[0][46] =  data[110]; buffer[0][47] =  data[111]; buffer[0][48] =  data[112]; buffer[0][49] =  data[113]; buffer[0][50] =  data[114]; buffer[0][51] =  data[115]; buffer[0][52] =  data[116]; buffer[0][53] =  data[117]; buffer[0][54] =  data[118]; buffer[0][55] =  data[119]; buffer[0][56] =  data[120]; buffer[0][57] =  data[121]; buffer[0][58] =  data[122]; buffer[0][59] =  data[123]; buffer[0][60] =  data[124]; buffer[0][61] =  data[125]; buffer[0][62] =  data[126]; buffer[0][63] =  data[127];

        }
        if (partition ==   2) {
            buffer[0][0] =  data[128]; buffer[0][1] =  data[129]; buffer[0][2] =  data[130]; buffer[0][3] =  data[131]; buffer[0][4] =  data[132]; buffer[0][5] =  data[133]; buffer[0][6] =  data[134]; buffer[0][7] =  data[135]; buffer[0][8] =  data[136]; buffer[0][9] =  data[137]; buffer[0][10] =  data[138]; buffer[0][11] =  data[139]; buffer[0][12] =  data[140]; buffer[0][13] =  data[141]; buffer[0][14] =  data[142]; buffer[0][15] =  data[143]; buffer[0][16] =  data[144]; buffer[0][17] =  data[145]; buffer[0][18] =  data[146]; buffer[0][19] =  data[147]; buffer[0][20] =  data[148]; buffer[0][21] =  data[149]; buffer[0][22] =  data[150]; buffer[0][23] =  data[151]; buffer[0][24] =  data[152]; buffer[0][25] =  data[153]; buffer[0][26] =  data[154]; buffer[0][27] =  data[155]; buffer[0][28] =  data[156]; buffer[0][29] =  data[157]; buffer[0][30] =  data[158]; buffer[0][31] =  data[159]; buffer[0][32] =  data[160]; buffer[0][33] =  data[161]; buffer[0][34] =  data[162]; buffer[0][35] =  data[163]; buffer[0][36] =  data[164]; buffer[0][37] =  data[165]; buffer[0][38] =  data[166]; buffer[0][39] =  data[167]; buffer[0][40] =  data[168]; buffer[0][41] =  data[169]; buffer[0][42] =  data[170]; buffer[0][43] =  data[171]; buffer[0][44] =  data[172]; buffer[0][45] =  data[173]; buffer[0][46] =  data[174]; buffer[0][47] =  data[175]; buffer[0][48] =  data[176]; buffer[0][49] =  data[177]; buffer[0][50] =  data[178]; buffer[0][51] =  data[179]; buffer[0][52] =  data[180]; buffer[0][53] =  data[181]; buffer[0][54] =  data[182]; buffer[0][55] =  data[183]; buffer[0][56] =  data[184]; buffer[0][57] =  data[185]; buffer[0][58] =  data[186]; buffer[0][59] =  data[187]; buffer[0][60] =  data[188]; buffer[0][61] =  data[189]; buffer[0][62] =  data[190]; buffer[0][63] =  data[191];

        }
        if (partition ==   3) {
            buffer[0][0] =  data[192]; buffer[0][1] =  data[193]; buffer[0][2] =  data[194]; buffer[0][3] =  data[195]; buffer[0][4] =  data[196]; buffer[0][5] =  data[197]; buffer[0][6] =  data[198]; buffer[0][7] =  data[199]; buffer[0][8] =  data[200]; buffer[0][9] =  data[201]; buffer[0][10] =  data[202]; buffer[0][11] =  data[203]; buffer[0][12] =  data[204]; buffer[0][13] =  data[205]; buffer[0][14] =  data[206]; buffer[0][15] =  data[207]; buffer[0][16] =  data[208]; buffer[0][17] =  data[209]; buffer[0][18] =  data[210]; buffer[0][19] =  data[211]; buffer[0][20] =  data[212]; buffer[0][21] =  data[213]; buffer[0][22] =  data[214]; buffer[0][23] =  data[215]; buffer[0][24] =  data[216]; buffer[0][25] =  data[217]; buffer[0][26] =  data[218]; buffer[0][27] =  data[219]; buffer[0][28] =  data[220]; buffer[0][29] =  data[221]; buffer[0][30] =  data[222]; buffer[0][31] =  data[223]; buffer[0][32] =  data[224]; buffer[0][33] =  data[225]; buffer[0][34] =  data[226]; buffer[0][35] =  data[227]; buffer[0][36] =  data[228]; buffer[0][37] =  data[229]; buffer[0][38] =  data[230]; buffer[0][39] =  data[231]; buffer[0][40] =  data[232]; buffer[0][41] =  data[233]; buffer[0][42] =  data[234]; buffer[0][43] =  data[235]; buffer[0][44] =  data[236]; buffer[0][45] =  data[237]; buffer[0][46] =  data[238]; buffer[0][47] =  data[239]; buffer[0][48] =  data[240]; buffer[0][49] =  data[241]; buffer[0][50] =  data[242]; buffer[0][51] =  data[243]; buffer[0][52] =  data[244]; buffer[0][53] =  data[245]; buffer[0][54] =  data[246]; buffer[0][55] =  data[247]; buffer[0][56] =  data[248]; buffer[0][57] =  data[249]; buffer[0][58] =  data[250]; buffer[0][59] =  data[251]; buffer[0][60] =  data[252]; buffer[0][61] =  data[253]; buffer[0][62] =  data[254]; buffer[0][63] =  data[255];

        }
        if (partition ==   4) {
            buffer[0][0] =  data[256]; buffer[0][1] =  data[257]; buffer[0][2] =  data[258]; buffer[0][3] =  data[259]; buffer[0][4] =  data[260]; buffer[0][5] =  data[261]; buffer[0][6] =  data[262]; buffer[0][7] =  data[263]; buffer[0][8] =  data[264]; buffer[0][9] =  data[265]; buffer[0][10] =  data[266]; buffer[0][11] =  data[267]; buffer[0][12] =  data[268]; buffer[0][13] =  data[269]; buffer[0][14] =  data[270]; buffer[0][15] =  data[271]; buffer[0][16] =  data[272]; buffer[0][17] =  data[273]; buffer[0][18] =  data[274]; buffer[0][19] =  data[275]; buffer[0][20] =  data[276]; buffer[0][21] =  data[277]; buffer[0][22] =  data[278]; buffer[0][23] =  data[279]; buffer[0][24] =  data[280]; buffer[0][25] =  data[281]; buffer[0][26] =  data[282]; buffer[0][27] =  data[283]; buffer[0][28] =  data[284]; buffer[0][29] =  data[285]; buffer[0][30] =  data[286]; buffer[0][31] =  data[287]; buffer[0][32] =  data[288]; buffer[0][33] =  data[289]; buffer[0][34] =  data[290]; buffer[0][35] =  data[291]; buffer[0][36] =  data[292]; buffer[0][37] =  data[293]; buffer[0][38] =  data[294]; buffer[0][39] =  data[295]; buffer[0][40] =  data[296]; buffer[0][41] =  data[297]; buffer[0][42] =  data[298]; buffer[0][43] =  data[299]; buffer[0][44] =  data[300]; buffer[0][45] =  data[301]; buffer[0][46] =  data[302]; buffer[0][47] =  data[303]; buffer[0][48] =  data[304]; buffer[0][49] =  data[305]; buffer[0][50] =  data[306]; buffer[0][51] =  data[307]; buffer[0][52] =  data[308]; buffer[0][53] =  data[309]; buffer[0][54] =  data[310]; buffer[0][55] =  data[311]; buffer[0][56] =  data[312]; buffer[0][57] =  data[313]; buffer[0][58] =  data[314]; buffer[0][59] =  data[315]; buffer[0][60] =  data[316]; buffer[0][61] =  data[317]; buffer[0][62] =  data[318]; buffer[0][63] =  data[319];

        }
        if (partition ==   5) {
            buffer[0][0] =  data[320]; buffer[0][1] =  data[321]; buffer[0][2] =  data[322]; buffer[0][3] =  data[323]; buffer[0][4] =  data[324]; buffer[0][5] =  data[325]; buffer[0][6] =  data[326]; buffer[0][7] =  data[327]; buffer[0][8] =  data[328]; buffer[0][9] =  data[329]; buffer[0][10] =  data[330]; buffer[0][11] =  data[331]; buffer[0][12] =  data[332]; buffer[0][13] =  data[333]; buffer[0][14] =  data[334]; buffer[0][15] =  data[335]; buffer[0][16] =  data[336]; buffer[0][17] =  data[337]; buffer[0][18] =  data[338]; buffer[0][19] =  data[339]; buffer[0][20] =  data[340]; buffer[0][21] =  data[341]; buffer[0][22] =  data[342]; buffer[0][23] =  data[343]; buffer[0][24] =  data[344]; buffer[0][25] =  data[345]; buffer[0][26] =  data[346]; buffer[0][27] =  data[347]; buffer[0][28] =  data[348]; buffer[0][29] =  data[349]; buffer[0][30] =  data[350]; buffer[0][31] =  data[351]; buffer[0][32] =  data[352]; buffer[0][33] =  data[353]; buffer[0][34] =  data[354]; buffer[0][35] =  data[355]; buffer[0][36] =  data[356]; buffer[0][37] =  data[357]; buffer[0][38] =  data[358]; buffer[0][39] =  data[359]; buffer[0][40] =  data[360]; buffer[0][41] =  data[361]; buffer[0][42] =  data[362]; buffer[0][43] =  data[363]; buffer[0][44] =  data[364]; buffer[0][45] =  data[365]; buffer[0][46] =  data[366]; buffer[0][47] =  data[367]; buffer[0][48] =  data[368]; buffer[0][49] =  data[369]; buffer[0][50] =  data[370]; buffer[0][51] =  data[371]; buffer[0][52] =  data[372]; buffer[0][53] =  data[373]; buffer[0][54] =  data[374]; buffer[0][55] =  data[375]; buffer[0][56] =  data[376]; buffer[0][57] =  data[377]; buffer[0][58] =  data[378]; buffer[0][59] =  data[379]; buffer[0][60] =  data[380]; buffer[0][61] =  data[381]; buffer[0][62] =  data[382]; buffer[0][63] =  data[383];

        }
        if (partition ==   6) {
            buffer[0][0] =  data[384]; buffer[0][1] =  data[385]; buffer[0][2] =  data[386]; buffer[0][3] =  data[387]; buffer[0][4] =  data[388]; buffer[0][5] =  data[389]; buffer[0][6] =  data[390]; buffer[0][7] =  data[391]; buffer[0][8] =  data[392]; buffer[0][9] =  data[393]; buffer[0][10] =  data[394]; buffer[0][11] =  data[395]; buffer[0][12] =  data[396]; buffer[0][13] =  data[397]; buffer[0][14] =  data[398]; buffer[0][15] =  data[399]; buffer[0][16] =  data[400]; buffer[0][17] =  data[401]; buffer[0][18] =  data[402]; buffer[0][19] =  data[403]; buffer[0][20] =  data[404]; buffer[0][21] =  data[405]; buffer[0][22] =  data[406]; buffer[0][23] =  data[407]; buffer[0][24] =  data[408]; buffer[0][25] =  data[409]; buffer[0][26] =  data[410]; buffer[0][27] =  data[411]; buffer[0][28] =  data[412]; buffer[0][29] =  data[413]; buffer[0][30] =  data[414]; buffer[0][31] =  data[415]; buffer[0][32] =  data[416]; buffer[0][33] =  data[417]; buffer[0][34] =  data[418]; buffer[0][35] =  data[419]; buffer[0][36] =  data[420]; buffer[0][37] =  data[421]; buffer[0][38] =  data[422]; buffer[0][39] =  data[423]; buffer[0][40] =  data[424]; buffer[0][41] =  data[425]; buffer[0][42] =  data[426]; buffer[0][43] =  data[427]; buffer[0][44] =  data[428]; buffer[0][45] =  data[429]; buffer[0][46] =  data[430]; buffer[0][47] =  data[431]; buffer[0][48] =  data[432]; buffer[0][49] =  data[433]; buffer[0][50] =  data[434]; buffer[0][51] =  data[435]; buffer[0][52] =  data[436]; buffer[0][53] =  data[437]; buffer[0][54] =  data[438]; buffer[0][55] =  data[439]; buffer[0][56] =  data[440]; buffer[0][57] =  data[441]; buffer[0][58] =  data[442]; buffer[0][59] =  data[443]; buffer[0][60] =  data[444]; buffer[0][61] =  data[445]; buffer[0][62] =  data[446]; buffer[0][63] =  data[447];

        }
        if (partition ==   7) {
            buffer[0][0] =  data[448]; buffer[0][1] =  data[449]; buffer[0][2] =  data[450]; buffer[0][3] =  data[451]; buffer[0][4] =  data[452]; buffer[0][5] =  data[453]; buffer[0][6] =  data[454]; buffer[0][7] =  data[455]; buffer[0][8] =  data[456]; buffer[0][9] =  data[457]; buffer[0][10] =  data[458]; buffer[0][11] =  data[459]; buffer[0][12] =  data[460]; buffer[0][13] =  data[461]; buffer[0][14] =  data[462]; buffer[0][15] =  data[463]; buffer[0][16] =  data[464]; buffer[0][17] =  data[465]; buffer[0][18] =  data[466]; buffer[0][19] =  data[467]; buffer[0][20] =  data[468]; buffer[0][21] =  data[469]; buffer[0][22] =  data[470]; buffer[0][23] =  data[471]; buffer[0][24] =  data[472]; buffer[0][25] =  data[473]; buffer[0][26] =  data[474]; buffer[0][27] =  data[475]; buffer[0][28] =  data[476]; buffer[0][29] =  data[477]; buffer[0][30] =  data[478]; buffer[0][31] =  data[479]; buffer[0][32] =  data[480]; buffer[0][33] =  data[481]; buffer[0][34] =  data[482]; buffer[0][35] =  data[483]; buffer[0][36] =  data[484]; buffer[0][37] =  data[485]; buffer[0][38] =  data[486]; buffer[0][39] =  data[487]; buffer[0][40] =  data[488]; buffer[0][41] =  data[489]; buffer[0][42] =  data[490]; buffer[0][43] =  data[491]; buffer[0][44] =  data[492]; buffer[0][45] =  data[493]; buffer[0][46] =  data[494]; buffer[0][47] =  data[495]; buffer[0][48] =  data[496]; buffer[0][49] =  data[497]; buffer[0][50] =  data[498]; buffer[0][51] =  data[499]; buffer[0][52] =  data[500]; buffer[0][53] =  data[501]; buffer[0][54] =  data[502]; buffer[0][55] =  data[503]; buffer[0][56] =  data[504]; buffer[0][57] =  data[505]; buffer[0][58] =  data[506]; buffer[0][59] =  data[507]; buffer[0][60] =  data[508]; buffer[0][61] =  data[509]; buffer[0][62] =  data[510]; buffer[0][63] =  data[511];

        }
    }
};
template<class data_T, class res_T, typename CONFIG_T>
class pointwise_conv_30 : public Conv1DKernel<data_T, res_T, CONFIG_T> {
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
template<class data_T, typename CONFIG_T>
class fill_buffer_31 : public nnet::FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9]; buffer[0][10] =   data[10]; buffer[0][11] =   data[11]; buffer[0][12] =   data[12]; buffer[0][13] =   data[13]; buffer[0][14] =   data[14]; buffer[0][15] =   data[15]; buffer[0][16] =   data[16]; buffer[0][17] =   data[17]; buffer[0][18] =   data[18]; buffer[0][19] =   data[19]; buffer[0][20] =   data[20]; buffer[0][21] =   data[21]; buffer[0][22] =   data[22]; buffer[0][23] =   data[23]; buffer[0][24] =   data[24]; buffer[0][25] =   data[25]; buffer[0][26] =   data[26]; buffer[0][27] =   data[27]; buffer[0][28] =   data[28]; buffer[0][29] =   data[29]; buffer[0][30] =   data[30]; buffer[0][31] =   data[31]; buffer[0][32] =   data[32]; buffer[0][33] =   data[33]; buffer[0][34] =   data[34]; buffer[0][35] =   data[35]; buffer[0][36] =   data[36]; buffer[0][37] =   data[37]; buffer[0][38] =   data[38]; buffer[0][39] =   data[39]; buffer[0][40] =   data[40]; buffer[0][41] =   data[41]; buffer[0][42] =   data[42]; buffer[0][43] =   data[43]; buffer[0][44] =   data[44]; buffer[0][45] =   data[45]; buffer[0][46] =   data[46]; buffer[0][47] =   data[47]; buffer[0][48] =   data[48]; buffer[0][49] =   data[49]; buffer[0][50] =   data[50]; buffer[0][51] =   data[51]; buffer[0][52] =   data[52]; buffer[0][53] =   data[53]; buffer[0][54] =   data[54]; buffer[0][55] =   data[55]; buffer[0][56] =   data[56]; buffer[0][57] =   data[57]; buffer[0][58] =   data[58]; buffer[0][59] =   data[59]; buffer[0][60] =   data[60]; buffer[0][61] =   data[61]; buffer[0][62] =   data[62]; buffer[0][63] =   data[63];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[64]; buffer[0][1] =   data[65]; buffer[0][2] =   data[66]; buffer[0][3] =   data[67]; buffer[0][4] =   data[68]; buffer[0][5] =   data[69]; buffer[0][6] =   data[70]; buffer[0][7] =   data[71]; buffer[0][8] =   data[72]; buffer[0][9] =   data[73]; buffer[0][10] =   data[74]; buffer[0][11] =   data[75]; buffer[0][12] =   data[76]; buffer[0][13] =   data[77]; buffer[0][14] =   data[78]; buffer[0][15] =   data[79]; buffer[0][16] =   data[80]; buffer[0][17] =   data[81]; buffer[0][18] =   data[82]; buffer[0][19] =   data[83]; buffer[0][20] =   data[84]; buffer[0][21] =   data[85]; buffer[0][22] =   data[86]; buffer[0][23] =   data[87]; buffer[0][24] =   data[88]; buffer[0][25] =   data[89]; buffer[0][26] =   data[90]; buffer[0][27] =   data[91]; buffer[0][28] =   data[92]; buffer[0][29] =   data[93]; buffer[0][30] =   data[94]; buffer[0][31] =   data[95]; buffer[0][32] =   data[96]; buffer[0][33] =   data[97]; buffer[0][34] =   data[98]; buffer[0][35] =   data[99]; buffer[0][36] =  data[100]; buffer[0][37] =  data[101]; buffer[0][38] =  data[102]; buffer[0][39] =  data[103]; buffer[0][40] =  data[104]; buffer[0][41] =  data[105]; buffer[0][42] =  data[106]; buffer[0][43] =  data[107]; buffer[0][44] =  data[108]; buffer[0][45] =  data[109]; buffer[0][46] =  data[110]; buffer[0][47] =  data[111]; buffer[0][48] =  data[112]; buffer[0][49] =  data[113]; buffer[0][50] =  data[114]; buffer[0][51] =  data[115]; buffer[0][52] =  data[116]; buffer[0][53] =  data[117]; buffer[0][54] =  data[118]; buffer[0][55] =  data[119]; buffer[0][56] =  data[120]; buffer[0][57] =  data[121]; buffer[0][58] =  data[122]; buffer[0][59] =  data[123]; buffer[0][60] =  data[124]; buffer[0][61] =  data[125]; buffer[0][62] =  data[126]; buffer[0][63] =  data[127];

        }
        if (partition ==   2) {
            buffer[0][0] =  data[128]; buffer[0][1] =  data[129]; buffer[0][2] =  data[130]; buffer[0][3] =  data[131]; buffer[0][4] =  data[132]; buffer[0][5] =  data[133]; buffer[0][6] =  data[134]; buffer[0][7] =  data[135]; buffer[0][8] =  data[136]; buffer[0][9] =  data[137]; buffer[0][10] =  data[138]; buffer[0][11] =  data[139]; buffer[0][12] =  data[140]; buffer[0][13] =  data[141]; buffer[0][14] =  data[142]; buffer[0][15] =  data[143]; buffer[0][16] =  data[144]; buffer[0][17] =  data[145]; buffer[0][18] =  data[146]; buffer[0][19] =  data[147]; buffer[0][20] =  data[148]; buffer[0][21] =  data[149]; buffer[0][22] =  data[150]; buffer[0][23] =  data[151]; buffer[0][24] =  data[152]; buffer[0][25] =  data[153]; buffer[0][26] =  data[154]; buffer[0][27] =  data[155]; buffer[0][28] =  data[156]; buffer[0][29] =  data[157]; buffer[0][30] =  data[158]; buffer[0][31] =  data[159]; buffer[0][32] =  data[160]; buffer[0][33] =  data[161]; buffer[0][34] =  data[162]; buffer[0][35] =  data[163]; buffer[0][36] =  data[164]; buffer[0][37] =  data[165]; buffer[0][38] =  data[166]; buffer[0][39] =  data[167]; buffer[0][40] =  data[168]; buffer[0][41] =  data[169]; buffer[0][42] =  data[170]; buffer[0][43] =  data[171]; buffer[0][44] =  data[172]; buffer[0][45] =  data[173]; buffer[0][46] =  data[174]; buffer[0][47] =  data[175]; buffer[0][48] =  data[176]; buffer[0][49] =  data[177]; buffer[0][50] =  data[178]; buffer[0][51] =  data[179]; buffer[0][52] =  data[180]; buffer[0][53] =  data[181]; buffer[0][54] =  data[182]; buffer[0][55] =  data[183]; buffer[0][56] =  data[184]; buffer[0][57] =  data[185]; buffer[0][58] =  data[186]; buffer[0][59] =  data[187]; buffer[0][60] =  data[188]; buffer[0][61] =  data[189]; buffer[0][62] =  data[190]; buffer[0][63] =  data[191];

        }
        if (partition ==   3) {
            buffer[0][0] =  data[192]; buffer[0][1] =  data[193]; buffer[0][2] =  data[194]; buffer[0][3] =  data[195]; buffer[0][4] =  data[196]; buffer[0][5] =  data[197]; buffer[0][6] =  data[198]; buffer[0][7] =  data[199]; buffer[0][8] =  data[200]; buffer[0][9] =  data[201]; buffer[0][10] =  data[202]; buffer[0][11] =  data[203]; buffer[0][12] =  data[204]; buffer[0][13] =  data[205]; buffer[0][14] =  data[206]; buffer[0][15] =  data[207]; buffer[0][16] =  data[208]; buffer[0][17] =  data[209]; buffer[0][18] =  data[210]; buffer[0][19] =  data[211]; buffer[0][20] =  data[212]; buffer[0][21] =  data[213]; buffer[0][22] =  data[214]; buffer[0][23] =  data[215]; buffer[0][24] =  data[216]; buffer[0][25] =  data[217]; buffer[0][26] =  data[218]; buffer[0][27] =  data[219]; buffer[0][28] =  data[220]; buffer[0][29] =  data[221]; buffer[0][30] =  data[222]; buffer[0][31] =  data[223]; buffer[0][32] =  data[224]; buffer[0][33] =  data[225]; buffer[0][34] =  data[226]; buffer[0][35] =  data[227]; buffer[0][36] =  data[228]; buffer[0][37] =  data[229]; buffer[0][38] =  data[230]; buffer[0][39] =  data[231]; buffer[0][40] =  data[232]; buffer[0][41] =  data[233]; buffer[0][42] =  data[234]; buffer[0][43] =  data[235]; buffer[0][44] =  data[236]; buffer[0][45] =  data[237]; buffer[0][46] =  data[238]; buffer[0][47] =  data[239]; buffer[0][48] =  data[240]; buffer[0][49] =  data[241]; buffer[0][50] =  data[242]; buffer[0][51] =  data[243]; buffer[0][52] =  data[244]; buffer[0][53] =  data[245]; buffer[0][54] =  data[246]; buffer[0][55] =  data[247]; buffer[0][56] =  data[248]; buffer[0][57] =  data[249]; buffer[0][58] =  data[250]; buffer[0][59] =  data[251]; buffer[0][60] =  data[252]; buffer[0][61] =  data[253]; buffer[0][62] =  data[254]; buffer[0][63] =  data[255];

        }
        if (partition ==   4) {
            buffer[0][0] =  data[256]; buffer[0][1] =  data[257]; buffer[0][2] =  data[258]; buffer[0][3] =  data[259]; buffer[0][4] =  data[260]; buffer[0][5] =  data[261]; buffer[0][6] =  data[262]; buffer[0][7] =  data[263]; buffer[0][8] =  data[264]; buffer[0][9] =  data[265]; buffer[0][10] =  data[266]; buffer[0][11] =  data[267]; buffer[0][12] =  data[268]; buffer[0][13] =  data[269]; buffer[0][14] =  data[270]; buffer[0][15] =  data[271]; buffer[0][16] =  data[272]; buffer[0][17] =  data[273]; buffer[0][18] =  data[274]; buffer[0][19] =  data[275]; buffer[0][20] =  data[276]; buffer[0][21] =  data[277]; buffer[0][22] =  data[278]; buffer[0][23] =  data[279]; buffer[0][24] =  data[280]; buffer[0][25] =  data[281]; buffer[0][26] =  data[282]; buffer[0][27] =  data[283]; buffer[0][28] =  data[284]; buffer[0][29] =  data[285]; buffer[0][30] =  data[286]; buffer[0][31] =  data[287]; buffer[0][32] =  data[288]; buffer[0][33] =  data[289]; buffer[0][34] =  data[290]; buffer[0][35] =  data[291]; buffer[0][36] =  data[292]; buffer[0][37] =  data[293]; buffer[0][38] =  data[294]; buffer[0][39] =  data[295]; buffer[0][40] =  data[296]; buffer[0][41] =  data[297]; buffer[0][42] =  data[298]; buffer[0][43] =  data[299]; buffer[0][44] =  data[300]; buffer[0][45] =  data[301]; buffer[0][46] =  data[302]; buffer[0][47] =  data[303]; buffer[0][48] =  data[304]; buffer[0][49] =  data[305]; buffer[0][50] =  data[306]; buffer[0][51] =  data[307]; buffer[0][52] =  data[308]; buffer[0][53] =  data[309]; buffer[0][54] =  data[310]; buffer[0][55] =  data[311]; buffer[0][56] =  data[312]; buffer[0][57] =  data[313]; buffer[0][58] =  data[314]; buffer[0][59] =  data[315]; buffer[0][60] =  data[316]; buffer[0][61] =  data[317]; buffer[0][62] =  data[318]; buffer[0][63] =  data[319];

        }
        if (partition ==   5) {
            buffer[0][0] =  data[320]; buffer[0][1] =  data[321]; buffer[0][2] =  data[322]; buffer[0][3] =  data[323]; buffer[0][4] =  data[324]; buffer[0][5] =  data[325]; buffer[0][6] =  data[326]; buffer[0][7] =  data[327]; buffer[0][8] =  data[328]; buffer[0][9] =  data[329]; buffer[0][10] =  data[330]; buffer[0][11] =  data[331]; buffer[0][12] =  data[332]; buffer[0][13] =  data[333]; buffer[0][14] =  data[334]; buffer[0][15] =  data[335]; buffer[0][16] =  data[336]; buffer[0][17] =  data[337]; buffer[0][18] =  data[338]; buffer[0][19] =  data[339]; buffer[0][20] =  data[340]; buffer[0][21] =  data[341]; buffer[0][22] =  data[342]; buffer[0][23] =  data[343]; buffer[0][24] =  data[344]; buffer[0][25] =  data[345]; buffer[0][26] =  data[346]; buffer[0][27] =  data[347]; buffer[0][28] =  data[348]; buffer[0][29] =  data[349]; buffer[0][30] =  data[350]; buffer[0][31] =  data[351]; buffer[0][32] =  data[352]; buffer[0][33] =  data[353]; buffer[0][34] =  data[354]; buffer[0][35] =  data[355]; buffer[0][36] =  data[356]; buffer[0][37] =  data[357]; buffer[0][38] =  data[358]; buffer[0][39] =  data[359]; buffer[0][40] =  data[360]; buffer[0][41] =  data[361]; buffer[0][42] =  data[362]; buffer[0][43] =  data[363]; buffer[0][44] =  data[364]; buffer[0][45] =  data[365]; buffer[0][46] =  data[366]; buffer[0][47] =  data[367]; buffer[0][48] =  data[368]; buffer[0][49] =  data[369]; buffer[0][50] =  data[370]; buffer[0][51] =  data[371]; buffer[0][52] =  data[372]; buffer[0][53] =  data[373]; buffer[0][54] =  data[374]; buffer[0][55] =  data[375]; buffer[0][56] =  data[376]; buffer[0][57] =  data[377]; buffer[0][58] =  data[378]; buffer[0][59] =  data[379]; buffer[0][60] =  data[380]; buffer[0][61] =  data[381]; buffer[0][62] =  data[382]; buffer[0][63] =  data[383];

        }
        if (partition ==   6) {
            buffer[0][0] =  data[384]; buffer[0][1] =  data[385]; buffer[0][2] =  data[386]; buffer[0][3] =  data[387]; buffer[0][4] =  data[388]; buffer[0][5] =  data[389]; buffer[0][6] =  data[390]; buffer[0][7] =  data[391]; buffer[0][8] =  data[392]; buffer[0][9] =  data[393]; buffer[0][10] =  data[394]; buffer[0][11] =  data[395]; buffer[0][12] =  data[396]; buffer[0][13] =  data[397]; buffer[0][14] =  data[398]; buffer[0][15] =  data[399]; buffer[0][16] =  data[400]; buffer[0][17] =  data[401]; buffer[0][18] =  data[402]; buffer[0][19] =  data[403]; buffer[0][20] =  data[404]; buffer[0][21] =  data[405]; buffer[0][22] =  data[406]; buffer[0][23] =  data[407]; buffer[0][24] =  data[408]; buffer[0][25] =  data[409]; buffer[0][26] =  data[410]; buffer[0][27] =  data[411]; buffer[0][28] =  data[412]; buffer[0][29] =  data[413]; buffer[0][30] =  data[414]; buffer[0][31] =  data[415]; buffer[0][32] =  data[416]; buffer[0][33] =  data[417]; buffer[0][34] =  data[418]; buffer[0][35] =  data[419]; buffer[0][36] =  data[420]; buffer[0][37] =  data[421]; buffer[0][38] =  data[422]; buffer[0][39] =  data[423]; buffer[0][40] =  data[424]; buffer[0][41] =  data[425]; buffer[0][42] =  data[426]; buffer[0][43] =  data[427]; buffer[0][44] =  data[428]; buffer[0][45] =  data[429]; buffer[0][46] =  data[430]; buffer[0][47] =  data[431]; buffer[0][48] =  data[432]; buffer[0][49] =  data[433]; buffer[0][50] =  data[434]; buffer[0][51] =  data[435]; buffer[0][52] =  data[436]; buffer[0][53] =  data[437]; buffer[0][54] =  data[438]; buffer[0][55] =  data[439]; buffer[0][56] =  data[440]; buffer[0][57] =  data[441]; buffer[0][58] =  data[442]; buffer[0][59] =  data[443]; buffer[0][60] =  data[444]; buffer[0][61] =  data[445]; buffer[0][62] =  data[446]; buffer[0][63] =  data[447];

        }
        if (partition ==   7) {
            buffer[0][0] =  data[448]; buffer[0][1] =  data[449]; buffer[0][2] =  data[450]; buffer[0][3] =  data[451]; buffer[0][4] =  data[452]; buffer[0][5] =  data[453]; buffer[0][6] =  data[454]; buffer[0][7] =  data[455]; buffer[0][8] =  data[456]; buffer[0][9] =  data[457]; buffer[0][10] =  data[458]; buffer[0][11] =  data[459]; buffer[0][12] =  data[460]; buffer[0][13] =  data[461]; buffer[0][14] =  data[462]; buffer[0][15] =  data[463]; buffer[0][16] =  data[464]; buffer[0][17] =  data[465]; buffer[0][18] =  data[466]; buffer[0][19] =  data[467]; buffer[0][20] =  data[468]; buffer[0][21] =  data[469]; buffer[0][22] =  data[470]; buffer[0][23] =  data[471]; buffer[0][24] =  data[472]; buffer[0][25] =  data[473]; buffer[0][26] =  data[474]; buffer[0][27] =  data[475]; buffer[0][28] =  data[476]; buffer[0][29] =  data[477]; buffer[0][30] =  data[478]; buffer[0][31] =  data[479]; buffer[0][32] =  data[480]; buffer[0][33] =  data[481]; buffer[0][34] =  data[482]; buffer[0][35] =  data[483]; buffer[0][36] =  data[484]; buffer[0][37] =  data[485]; buffer[0][38] =  data[486]; buffer[0][39] =  data[487]; buffer[0][40] =  data[488]; buffer[0][41] =  data[489]; buffer[0][42] =  data[490]; buffer[0][43] =  data[491]; buffer[0][44] =  data[492]; buffer[0][45] =  data[493]; buffer[0][46] =  data[494]; buffer[0][47] =  data[495]; buffer[0][48] =  data[496]; buffer[0][49] =  data[497]; buffer[0][50] =  data[498]; buffer[0][51] =  data[499]; buffer[0][52] =  data[500]; buffer[0][53] =  data[501]; buffer[0][54] =  data[502]; buffer[0][55] =  data[503]; buffer[0][56] =  data[504]; buffer[0][57] =  data[505]; buffer[0][58] =  data[506]; buffer[0][59] =  data[507]; buffer[0][60] =  data[508]; buffer[0][61] =  data[509]; buffer[0][62] =  data[510]; buffer[0][63] =  data[511];

        }
    }
};
template<class data_T, class res_T, typename CONFIG_T>
class pointwise_conv_31 : public Conv1DKernel<data_T, res_T, CONFIG_T> {
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
