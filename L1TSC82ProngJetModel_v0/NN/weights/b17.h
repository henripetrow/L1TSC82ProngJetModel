//Numpy array shape [4]
//Min -0.125000000000
//Max 0.000000000000
//Number of zeros 1

#ifndef B17_H_
#define B17_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
bias17_t b17[4];
#else
bias17_t b17[4] = {0.0000000, -0.0703125, -0.1250000, -0.0156250};

#endif

#endif
