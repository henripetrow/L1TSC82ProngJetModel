//Numpy array shape [4]
//Min -0.143927127123
//Max 0.071971088648
//Number of zeros 0

#ifndef B28_H_
#define B28_H_

#ifdef __HLS4ML_LOAD_TXT_WEIGHTS__
phi3_bias_t b28[4];
#else
phi3_bias_t b28[4] = {-0.0368876, 0.0719711, -0.1439271, -0.0808001};

#endif

#endif
