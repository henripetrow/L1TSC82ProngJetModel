#include "NN/L1TSC82ProngJetModel_v0.h" //include of the top level of HLS model
#include "emulator.h" //include of emulator modeling
#include "NN/nnet_utils/nnet_common.h"
#include <any>
#include <array>
#include <utility>
#include "ap_fixed.h"
#include "ap_int.h"

using namespace hls4ml_L1TSC82ProngJetModel_v0;

class L1TSC82ProngJetModel_emulator_v0 : public hls4mlEmulator::Model{
    private:
        input_t _input[N_INPUT_1_1*N_INPUT_2_1];
        result_t _layer16_out[N_LAYER_14]; // 2-prong score, 1-prong score.
    public:


        virtual void prepare_input(std::any input)
        {
            input_t* input_p = std::any_cast<input_t*>(input);
            for(int i = 0; i < N_INPUT_1_1*N_INPUT_2_1; ++i){
                _input[i] = std::any_cast<input_t>(input_p[i]);
            }
        }



        virtual void predict()
        {
            L1TSC82ProngJetModel_v0(_input, _layer16_out);  
        }

        virtual void read_result(std::any result)
        { 
            result_t *result_p = std::any_cast<result_t*>(result);
            for (int i = 0; i < N_LAYER_14; i++) {
                result_p[i] = _layer16_out[i];
            }
        }

};

extern "C" hls4mlEmulator::Model* create_model()
{
    return new L1TSC82ProngJetModel_emulator_v0;
}

extern "C" void destroy_model(hls4mlEmulator::Model* m)
{
    delete m;
}
