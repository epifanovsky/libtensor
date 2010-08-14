#ifndef LIBTENSOR_LINALG_IMPL_MKL_H
#define LIBTENSOR_LINALG_IMPL_MKL_H

#include "linalg_impl_cblas.h"

namespace libtensor {


/**	\brief Implementation of linear algebra using Intel Math Kernel Library

	\sa linalg_impl_generic, linalg_impl_cblas

	\ingroup libtensor_linalg
 **/
class linalg_impl_mkl : public linalg_impl_cblas {
public:
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IMPL_MKL_H
