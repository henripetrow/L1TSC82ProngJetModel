//Numpy array shape [4]
//Min -0.236609131098
//Max 0.092827521265
//Number of zeros 0

#ifndef B28_H_
#define B28_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
phi3_bias_t b28[4];
#else
phi3_bias_t b28[4] = {-0.2366091, 0.0928275, -0.1122782, 0.0582628};

#endif

#endif
