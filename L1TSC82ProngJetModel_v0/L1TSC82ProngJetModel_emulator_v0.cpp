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
            input_t _input_layer[8*20];
            result_t _layer25_out[1];
    public:


        virtual void prepare_input(std::any input)
        {
            input_t *input_p = std::any_cast<input_t*>(input);
            for(int i = 0; i < 8*20; ++i){
                _input_layer[i] = std::any_cast<input_t>(input_p[i]);
            }
        }



        virtual void predict()
        {
            L1TSC82ProngJetModel_v0(_input_layer, _layer25_out);
            
        }

        virtual void read_result(std::any result)
        { 
            result_t *result_p = std::any_cast<result_t*>(result);
            for (int i = 0; i < 1; ++i ){
                result_p[i] = _layer25_out[i];  
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