#ifndef LIBTENSOR_LINALG1_H
#define LIBTENSOR_LINALG1_H

/** \defgroup libtensor_linalg Wrappers for linear algebra primitives
    \ingroup libtensor
 **/


#ifdef USE_MKL
#include "linalg/lapack_mkl.h"
#else // USE_MKL
#ifdef USE_ACML
#include "linalg/lapack_acml.h"
#else // USE_ACML
#ifdef USE_ESSL
#include "linalg/lapack_essl.h"
#else // USE_ESSL
#include "linalg/lapack_generic.h"
#endif // USE_ESSL
#endif // USE_ACML
#endif // USE_MKL

#endif // LIBTENSOR_LINALG1_H
