#ifndef LIBTENSOR_LINALG_BASE_LEVEL6_ADAPTIVE_H
#define LIBTENSOR_LINALG_BASE_LEVEL6_ADAPTIVE_H

#include "../generic/linalg_base_level6_generic.h"

namespace libtensor {


/**	\brief Level-6 linear algebra operations (adaptive)
	\tparam M Memory driver.
	\tparam L1 Level-1 driver.
	\tparam L2 Level-2 driver.
	\tparam L3 Level-3 driver.

	\ingroup libtensor_linalg
 **/
template<typename M, typename L1, typename L2, typename L3>
struct linalg_base_level6_adaptive : public linalg_base_level6_generic {


	static void ijkl_ipkq_pljq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_iplq_kpjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_iplq_pkjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_iplq_pkqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_ipqk_pljq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_ipql_pkjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_ipql_pkqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pikq_pljq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pilq_kpjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pilq_pkjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_piqk_pljq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_piql_kpqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_piql_pkjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_piql_pkqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pkiq_jplq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pkiq_jpql_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pkiq_pjlq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pkiq_pjql_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pliq_jpkq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pliq_jpqk_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	static void ijkl_pliq_pjqk_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);

};


} // namespace libtensor


#include "bits/ijkl_ipkq_pljq_x.h"
#include "bits/ijkl_iplq_kpjq_x.h"
#include "bits/ijkl_iplq_pkjq_x.h"
#include "bits/ijkl_iplq_pkqj_x.h"
#include "bits/ijkl_ipqk_pljq_x.h"
#include "bits/ijkl_ipql_pkjq_x.h"
#include "bits/ijkl_ipql_pkqj_x.h"
#include "bits/ijkl_pikq_pljq_x.h"
#include "bits/ijkl_pilq_kpjq_x.h"
#include "bits/ijkl_pilq_pkjq_x.h"
#include "bits/ijkl_piqk_pljq_x.h"
#include "bits/ijkl_piql_kpqj_x.h"
#include "bits/ijkl_piql_pkjq_x.h"
#include "bits/ijkl_piql_pkqj_x.h"
#include "bits/ijkl_pkiq_jplq_x.h"
#include "bits/ijkl_pkiq_jpql_x.h"
#include "bits/ijkl_pkiq_pjlq_x.h"
#include "bits/ijkl_pkiq_pjql_x.h"
#include "bits/ijkl_pliq_jpkq_x.h"
#include "bits/ijkl_pliq_jpqk_x.h"
#include "bits/ijkl_pliq_pjqk_x.h"


#endif // LIBTENSOR_LINALG_BASE_LEVEL6_ADAPTIVE_H
