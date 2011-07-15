#ifndef LIBTENSOR_LINALG_BASE_LEVEL3X_ADAPTIVE_H
#define LIBTENSOR_LINALG_BASE_LEVEL3X_ADAPTIVE_H

#include "../generic/linalg_base_level3x_generic.h"

namespace libtensor {


/**	\brief Level-3-extension linear algebra operations (adaptive)
	\tparam M Memory driver.
	\tparam L1 Level-1 driver.
	\tparam L2 Level-2 driver.

	\ingroup libtensor_linalg
 **/
template<typename M, typename L1, typename L2>
struct linalg_base_level3x_adaptive : public linalg_base_level3x_generic {


	static void ij_pji_p_x(
		size_t ni, size_t nj, size_t np,
		const double *a, size_t sja, size_t spa,
		const double *b, size_t spb,
		double *c, size_t sic,
		double d);


};


} // namespace libtensor


#include "bits/ij_pji_p_x.h"


#endif // LIBTENSOR_LINALG_BASE_LEVEL3X_ADAPTIVE_H
