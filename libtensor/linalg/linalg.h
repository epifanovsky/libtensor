#ifndef LIBTENSOR_LINALG_H
#define LIBTENSOR_LINALG_H

#if defined(USE_ACML)
#include "acml/linalg_acml.h"
namespace libtensor {
typedef linalg_acml linalg;
} // namespace libtensor

#elif defined(USE_CBLAS)
#include "cblas/linalg_cblas.h"
namespace libtensor {
typedef linalg_cblas linalg;
} // namespace libtensor

#elif defined(USE_ESSL)
#include "essl/linalg_essl.h"
namespace libtensor {
typedef linalg_essl linalg;
} // namespace libtensor

#elif defined(USE_GSL)
#include "gsl/linalg_gsl.h"
namespace libtensor {
typedef linalg_gsl linalg;
} // namespace libtensor

#elif defined(USE_MKL)
#include "mkl/linalg_mkl.h"
namespace libtensor {
typedef linalg_mkl linalg;
} // namespace libtensor

#elif defined(USE_QCHEM)
#include "qchem/linalg_qchem.h"
namespace libtensor {
typedef linalg_qchem linalg;
} // namespace libtensor

#else
#include "generic/linalg_generic.h"
namespace libtensor {
typedef linalg_generic linalg;
} // namespace libtensor

#endif

#endif // LIBTENSOR_LINALG_H
