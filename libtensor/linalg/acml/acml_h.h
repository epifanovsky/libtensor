#ifndef LIBTENSOR_ACML_H_H
#define LIBTENSOR_ACML_H_H


#include <complex>

#ifndef _ACML_COMPLEX
#define _ACML_COMPLEX 1
typedef std::complex<float> complex;
typedef std::complex<double> doublecomplex;
#endif // _ACML_COMPLEX

#include <acml.h>

#endif // LIBTENSOR_ACML_H_H
