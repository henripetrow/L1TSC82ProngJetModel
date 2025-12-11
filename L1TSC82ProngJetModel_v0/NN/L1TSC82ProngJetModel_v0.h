#ifndef L1TSC82PRONGJETMODEL_V0_H_
#define L1TSC82PRONGJETMODEL_V0_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
namespace hls4ml_L1TSC82ProngJetModel_v0 {
void L1TSC82ProngJetModel_v0(
    input_t input_layer[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer25_out[N_LAYER_23]
);


} // namespace hls4ml_L1TSC82ProngJetModel_v0
#endif
