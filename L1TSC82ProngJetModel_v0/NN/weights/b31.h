//Numpy array shape [4]
//Min -0.181659936905
//Max 0.184647813439
//Number of zeros 0

#ifndef B31_H_
#define B31_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
phi3_bias_t b31[4];
#else
phi3_bias_t b31[4] = {0.1846478, -0.1816599, -0.1287615, 0.0459674};

#endif

#endif
