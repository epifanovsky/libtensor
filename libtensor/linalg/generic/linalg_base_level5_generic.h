#ifndef LIBTENSOR_LINALG_BASE_LEVEL5_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL5_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/**	\brief Level-5 linear algebra operations (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level5_generic {


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{p} a_{ipl} b_{kpj} d \f$
	 **/
	static void ijkl_ipl_kpj_x(
		size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		const double *a, const double *b, double *c, double d);


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL5_GENERIC_H
