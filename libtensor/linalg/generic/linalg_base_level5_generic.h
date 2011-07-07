#ifndef LIBTENSOR_LINALG_BASE_LEVEL5_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL5_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/**	\brief Level-5 linear algebra operations (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level5_generic {


	/**	\brief \f$ c_{ijk} = c_{ijk} +
			\sum_{pq} a_{ipq} b_{kjqp} d \f$
	 **/
	static void ijk_ipq_kjqp_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijk} = c_{ijk} +
			\sum_{p} a_{ipkq} b_{jpq} d \f$
	 **/
	static void ijk_ipkq_jpq_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijk} = c_{ijk} +
			\sum_{p} a_{pikq} b_{jpq} d \f$
	 **/
	static void ijk_pikq_jpq_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijk} = c_{ijk} +
			\sum_{p} a_{piqk} b_{jpq} d \f$
	 **/
	static void ijk_piqk_jpq_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijk} = c_{ijk} +
			\sum_{p} a_{pkiq} b_{pjq} d \f$
	 **/
	static void ijk_pkiq_pjq_x(
		size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{p} a_{ipj} b_{plk} d \f$
	 **/
	static void ijkl_ipj_plk_x(
		size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		const double *a, const double *b, double *c, double d);


	/**	\brief \f$ c_{ijkl} = c_{ijkl} +
			\sum_{p} a_{ipl} b_{kpj} d \f$
	 **/
	static void ijkl_ipl_kpj_x(
		size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		const double *a, const double *b, double *c, double d);


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL5_GENERIC_H
