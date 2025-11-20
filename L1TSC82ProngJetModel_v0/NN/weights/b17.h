//Numpy array shape [4]
//Min -0.101562500000
//Max 0.093750000000
//Number of zeros 0

#ifndef B17_H_
#define B17_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
bias17_t b17[4];
#else
bias17_t b17[4] = {-0.0625000, 0.0937500, -0.1015625, 0.0078125};

#endif

#endif
