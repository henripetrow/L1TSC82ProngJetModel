#ifndef L1TSC8NGJETMODEL_H_
#define L1TSC8NGJETMODEL_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void L1TSC8NGJetModel(
    input_t input_layer[8*20],
    result_t layer25_out[1]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
