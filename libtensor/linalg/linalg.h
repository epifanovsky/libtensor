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

#endif // LIBTENSOR_LINALG_H
