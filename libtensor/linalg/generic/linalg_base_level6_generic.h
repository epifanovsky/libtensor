#ifndef LIBTENSOR_LINALG_BASE_LEVEL6_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL6_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/**	\brief Level-6 linear algebra operations (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level6_generic {


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{iplq} b_{kpjq} d \f$
	 **/
	static void ijkl_iplq_kpjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{iplq} b_{pkjq} d \f$
	 **/
	static void ijkl_iplq_pkjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{iplq} b_{pkqj} d \f$
	 **/
	static void ijkl_iplq_pkqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{ipql} b_{pkqj} d \f$
	 **/
	static void ijkl_ipql_pkqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pilq} b_{kpjq} d \f$
	 **/
	static void ijkl_pilq_kpjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pilq} b_{pkjq} d \f$
	 **/
	static void ijkl_pilq_pkjq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{piql} b_{kpqj} d \f$
	 **/
	static void ijkl_piql_kpqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{piql} b_{pkqj} d \f$
	 **/
	static void ijkl_piql_pkqj_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pkiq} b_{jplq} d \f$
	 **/
	static void ijkl_pkiq_jplq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pkiq} b_{jpql} d \f$
	 **/
	static void ijkl_pkiq_jpql_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pkiq} b_{pjlq} d \f$
	 **/
	static void ijkl_pkiq_pjlq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pkiq} b_{pjql} d \f$
	 **/
	static void ijkl_pkiq_pjql_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pliq} b_{jpkq} d \f$
	 **/
	static void ijkl_pliq_jpkq_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pliq} b_{jpqk} d \f$
	 **/
	static void ijkl_pliq_jpqk_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{pq} a_{pliq} b_{pjqk} d \f$
	 **/
	static void ijkl_pliq_pjqk_x(
		size_t ni, size_t nj, size_t nk,
		size_t nl, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL6_GENERIC_H
