//Numpy array shape [16, 1]
//Min -0.734375000000
//Max 1.046875000000
//Number of zeros 0

#ifndef W23_H_
#define W23_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
weight23_t w23[16];
#else
weight23_t ww23[16] = {0.406250, -0.734375, -0.531250, 0.375000, 0.375000, -0.437500, -0.406250, 0.640625, 1.046875, 0.437500, -0.500000, 0.703125, 0.546875, -0.703125, 0.890625, -0.656250};

#endif

#endif
