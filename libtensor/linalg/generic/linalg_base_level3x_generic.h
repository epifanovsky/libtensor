#ifndef LIBTENSOR_LINALG_BASE_LEVEL3X_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL3X_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/**	\brief Level-3-extension linear algebra operations (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level3x_generic {


	/**	\brief \f$ c_{ij} = c_{ij} + \sum_p a_{pji} b_{p} d \f$
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param a Pointer to a.
		\param sja Step of j in a (sja >= ni).
		\param spa Step of p in a (spa >= ni * nj).
		\param b Pointer to b.
		\param spb Step of p in b.
		\param c Pointer to c.
		\param sic Step of i in c (sic >= nj).
		\param d Value of d.
	 **/
	static void ij_pji_p_x(
		size_t ni, size_t nj, size_t np,
		const double *a, size_t sja, size_t spa,
		const double *b, size_t spb,
		double *c, size_t sic,
		double d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL3X_GENERIC_H
