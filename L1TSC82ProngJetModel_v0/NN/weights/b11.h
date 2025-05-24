//Numpy array shape [4]
//Min -0.031250000000
//Max 0.023437500000
//Number of zeros 0

#ifndef B11_H_
#define B11_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
bias11_t b11[4];
#else
bias11_t b11[4] = {-0.0234375, -0.0312500, 0.0234375, -0.0078125};

#endif

#endif
