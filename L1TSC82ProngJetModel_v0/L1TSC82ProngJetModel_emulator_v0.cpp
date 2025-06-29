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
        result_t _layer16_out[N_LAYER_14];
    public:


        virtual void prepare_input(std::any input)
        {
            input_t* input_p = std::any_cast<input_t*>(input);
            for(int i = 0; i < N_INPUT_1_1*N_INPUT_2_1; ++i){
                 _input[i] = std::any_cast<input_t>(input_p[i]);
                std::cout << _input[i];
                std::cout << ",";
            }
        }

        virtual void predict()
        {
            L1TSC82ProngJetModel_v0(_input, _layer16_out);  
        }

        virtual void read_result(std::any result)
        { 
            std::array<result_t, 1> *result_p = std::any_cast<std::array<result_t, 1>*>(result);
            (*result_p)[0] = _layer16_out[1];
            std::cout << _layer16_out[0];
            std::cout << ",";
            std::cout << _layer16_out[1];
            std::cout << "\n";
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
