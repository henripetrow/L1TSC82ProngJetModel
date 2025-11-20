//Numpy array shape [4]
//Min -0.046875000000
//Max 0.164062500000
//Number of zeros 1

#ifndef B20_H_
#define B20_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
bias20_t b20[4];
#else
bias20_t b20[4] = {-0.0468750, 0.0156250, 0.0000000, 0.1640625};

#endif

#endif
