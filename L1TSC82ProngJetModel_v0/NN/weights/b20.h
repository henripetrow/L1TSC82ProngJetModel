//Numpy array shape [4]
//Min -0.148437500000
//Max 0.171875000000
//Number of zeros 1

#ifndef B20_H_
#define B20_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
bias20_t b20[4];
#else
bias20_t b20[4] = {0.0000000, -0.1484375, -0.0234375, 0.1718750};

#endif

#endif
