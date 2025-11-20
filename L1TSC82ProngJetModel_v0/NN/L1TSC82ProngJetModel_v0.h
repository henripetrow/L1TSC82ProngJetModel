#ifndef L1TSC82PRONGJETMODEL_V0_H_
#define L1TSC82PRONGJETMODEL_V0_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

namespace hls4ml_L1TSC82ProngJetModel_v0 {
// Prototype of top level function for C-synthesis
void L1TSC82ProngJetModel_v0(
    input_t input_layer[8*20],
    result_t layer25_out[1]
);

// hls-fpga-machine-learning insert emulator-defines

} //namespace hls4ml_L1TSC82ProngJetModel_v0 

#endif
