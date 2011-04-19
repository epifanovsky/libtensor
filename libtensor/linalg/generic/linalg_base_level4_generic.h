#ifndef LIBTENSOR_LINALG_BASE_LEVEL4_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL4_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/**	\brief Level-4 linear algebra operations (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level4_generic {


	/**	\brief \f$ c_{ij} = \sum_{pq} a_{ipq} b_{jqp} d \f$
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param nq Number of elements q.
		\param a Pointer to a.
		\param spa Step of p in a (spa >= nq).
		\param sia Step of i in a (sia >= np * spa).
		\param b Pointer to b.
		\param sqb Step of q in b (sqb >= np).
		\param sjb Step of j in b (sjb >= nq * sqb).
		\param c Pointer to c.
		\param sic Step of i in c (sic >= ni).
		\param d Value of d.
	 **/
	static void ij_ipq_jqp_x(
		size_t ni, size_t nj, size_t np, size_t nq,
		const double *a, size_t spa, size_t sia,
		const double *b, size_t sqb, size_t sjb,
		double *c, size_t sic,
		double d);


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL4_GENERIC_H
