#ifndef LIBTENSOR_LINALG_BASE_LEVEL5_ADAPTIVE_H
#define LIBTENSOR_LINALG_BASE_LEVEL5_ADAPTIVE_H

#include "../generic/linalg_base_level5_generic.h"

namespace libtensor {


/**	\brief Level-5 linear algebra operations (adaptive)
	\tparam M Memory driver.
	\tparam L1 Level-1 driver.
	\tparam L2 Level-2 driver.
	\tparam L3 Level-3 driver.

	\ingroup libtensor_linalg
 **/
template<typename M, typename L1, typename L2, typename L3>
struct linalg_base_level5_adaptive : public linalg_base_level5_generic {


	static void ijk_ipq_kjqp_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijk_ipkq_jpq_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijk_pikq_jpq_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijk_piqk_jpq_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


};


} // namespace libtensor


#include "bits/ijk_ipq_kjqp_x.h"
#include "bits/ijk_ipkq_jpq_x.h"
#include "bits/ijk_pikq_jpq_x.h"
#include "bits/ijk_piqk_jpq_x.h"


#endif // LIBTENSOR_LINALG_BASE_LEVEL5_ADAPTIVE_H
