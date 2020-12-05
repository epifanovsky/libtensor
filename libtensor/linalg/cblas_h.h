#ifndef LIBTENSOR_CBLAS_H_H
#define LIBTENSOR_CBLAS_H_H

extern "C" {  // Fixes older cblas.h versions without extern "C"
#include <cblas.h>
}

#ifdef USE_GSL
#include <gsl/gsl_cblas.h>
#endif  // USE_GSL

#endif  // LIBTENSOR_CBLAS_H_H
