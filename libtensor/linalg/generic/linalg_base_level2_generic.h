#ifndef LIBTENSOR_LINALG_BASE_LEVEL2_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL2_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/**	\brief Level-2 linear algebra operations (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level2_generic {


	/**	\brief \f$ c_i = c_i + \sum_p a_{ip} b_p d \f$
		\param ni Number of elements i.
		\param np Number of elements p.
		\param a Pointer to a.
		\param sia Step of i in a (sia >= np).
		\param b Pointer to b.
		\param spb Step of p in b (spb >= 1).
		\param c Pointer to c.
		\param sic Step of i in c (sic >= 1).
		\param d Scalar d.
	 **/
	static void i_ip_p_x(
		size_t ni, size_t np,
		const double *a, size_t sia,
		const double *b, size_t spb,
		double *c, size_t sic,
		double d);


	/**	\brief \f$ c_i = c_i + \sum_p a_{pi} b_p d \f$
		\param ni Number of elements i.
		\param np Number of elements p.
		\param a Pointer to a.
		\param spa Step of p in a (spa >= ni).
		\param b Pointer to b.
		\param spb Step of p in b (spb >= 1).
		\param c Pointer to c.
		\param sic Step of i in c (sic >= 1).
		\param d Scalar d.
	 **/
	static void i_pi_p_x(
		size_t ni, size_t np,
		const double *a, size_t spa,
		const double *b, size_t spb,
		double *c, size_t sic,
		double d);


	/**	\brief \f$ c_{ij} = c_{ij} + a_i b_j d \f$
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param a Pointer to a.
		\param sia Step of i in a (sia >= 1).
		\param b Pointer to b.
		\param sjb Step of j in b (sjb >= 1).
		\param c Pointer to c.
		\param sic Step of i in c (sic >= nj).
		\param d Scalar d.
	 **/
	static void ij_i_j_x(
		size_t ni, size_t nj,
		const double *a, size_t sia,
		const double *b, size_t sjb,
		double *c, size_t sic,
		double d);


	/**	\brief \f$ c_{ij} = c_{ij} + a_{ji} b \f$
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param a Pointer to a.
		\param sja Step of j in a.
		\param b Scalar b.
		\param c Pointer to c.
		\param sic Step of i in c.
	 **/
	static void ij_ji_x(
		size_t ni, size_t nj,
		const double *a, size_t sja,
		double b,
		double *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL2_GENERIC_H
