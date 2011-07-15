#ifndef LIBTENSOR_LINALG_BASE_LEVEL4_ADAPTIVE_H
#define LIBTENSOR_LINALG_BASE_LEVEL4_ADAPTIVE_H

#include "../generic/linalg_base_level4_generic.h"

namespace libtensor {


/**	\brief Level-4 linear algebra operations (adaptive)
	\tparam M Memory driver.
	\tparam L1 Level-1 driver.
	\tparam L2 Level-2 driver.
	\tparam L3 Level-3 driver.

	\ingroup libtensor_linalg
 **/
template<typename M, typename L1, typename L2, typename L3>
struct linalg_base_level4_adaptive : public linalg_base_level4_generic {


	static void ij_ipq_jqp_x(
		size_t ni, size_t nj, size_t np, size_t nq,
		const double *a, size_t spa, size_t sia,
		const double *b, size_t sqb, size_t sjb,
		double *c, size_t sic,
		double d);


	static void ijk_ip_pkj_x(
		size_t ni, size_t nj, size_t nk, size_t np,
		const double *a, size_t sia,
		const double *b, size_t skb, size_t spb,
		double *c, size_t sjc, size_t sic,
		double d);


	static void ijk_pi_pkj_x(
		size_t ni, size_t nj, size_t nk, size_t np,
		const double *a, size_t spa,
		const double *b, size_t skb, size_t spb,
		double *c, size_t sjc, size_t sic,
		double d);


};


} // namespace libtensor


#include "bits/ij_ipq_jqp_x.h"
#include "bits/ijk_ip_pkj_x.h"
#include "bits/ijk_pi_pkj_x.h"


#endif // LIBTENSOR_LINALG_BASE_LEVEL4_ADAPTIVE_H
