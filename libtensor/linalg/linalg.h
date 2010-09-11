#ifndef LIBTENSOR_LINALG_H
#define LIBTENSOR_LINALG_H

#include "linalg_impl_generic.h"
#include "linalg_impl_cblas.h"
#include "linalg_impl_acml.h"
#include "linalg_impl_mkl.h"

namespace libtensor {


/**	\brief Provides low-level linear algebra routines

	\sa linalg_impl_generic

	\ingroup libtensor_linalg
 **/
class linalg :
#if defined(USE_MKL)
	public linalg_impl_mkl
#elif defined(USE_ACML)
	public linalg_impl_acml
#elif defined(USE_CBLAS) || defined(USE_GSL)
	public linalg_impl_cblas
#elif defined(USE_QCHEM)
	public linalg_impl_qchem
#else
	public linalg_impl_generic
#endif
{ };

} // namespace libtensor


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

struct linalg2 : public linalg_base { };

} // namespace libtensor


#endif // LIBTENSOR_LINALG_H
