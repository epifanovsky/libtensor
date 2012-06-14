#ifndef LIBTENSOR_LINALG_H
#define LIBTENSOR_LINALG_H

#include "generic/linalg_base_generic.h"

#if defined(USE_MKL)
#include "mkl/linalg_base_mkl.h"
namespace libtensor {
typedef linalg_base_mkl linalg_base;
} // namespace libtensor

#elif defined(USE_ACML)
#include "acml/linalg_base_acml.h"
namespace libtensor {
typedef linalg_base_acml linalg_base;
} // namespace libtensor

#elif defined(USE_ESSL)
#include "essl/linalg_base_essl.h"
namespace libtensor {
typedef linalg_base_essl linalg_base;
} // namespace libtensor

#elif defined(USE_GSL)
#include "gsl/linalg_base_gsl.h"
namespace libtensor {
typedef linalg_base_gsl linalg_base;
} // namespace libtensor

#elif defined(USE_CBLAS)
#include "cblas/linalg_base_cblas.h"
namespace libtensor {
typedef linalg_base_cblas linalg_base;
} // namespace libtensor

#elif defined(USE_QCHEM)
#include "qchem/linalg_base_qchem.h"
namespace libtensor {
typedef linalg_base_qchem linalg_base;
} // namespace libtensor

#else
namespace libtensor {
typedef linalg_base_generic linalg_base;
} // namespace libtensor

#endif


namespace libtensor {

/** \brief Provides basic linear algebra routines

    \ingroup libtensor_linalg
 **/
struct linalg : public linalg_base { };

} // namespace libtensor


#endif // LIBTENSOR_LINALG_H
