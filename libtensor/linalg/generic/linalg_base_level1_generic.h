#ifndef LIBTENSOR_LINALG_BASE_LEVEL1_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL1_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/**	\brief Level-1 linear algebra operations (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level1_generic {

	/**	\brief \f$ c_i = c_i + (a_i + b) d \f$
		\param ni Number of elements i.
		\param a Pointer to a.
		\param sia Step of i in a.
		\param b Scalar b.
		\param c Pointer to c.
		\param sic Step of i in c.
		\param d Scalar d.
	 **/
	static void add_i_i_x_x(
		size_t ni,
		const double *a, size_t sia, double ka,
		double b, double kb,
		double *c, size_t sic,
		double d);

	/**	\brief \f$ c_i = a_i \f$
		\param ni Number of elements i.
		\param a Scalar a.
		\param sia Step of i in a.
		\param c Pointer to c.
		\param sic Step of i in c.
	 **/
	static void i_i(
		size_t ni,
		const double *a, size_t sia,
		double *c, size_t sic);

	/**	\brief \f$ c_i = c_i a \f$
		\param ni Number of elements i.
		\param a Scalar a.
		\param c Pointer to c.
		\param sic Step of i in c.
	 **/
	static void i_x(
		size_t ni,
		double a,
		double *c, size_t sic);

	/**	\brief \f$ c = \sum_p a_p b_p \f$
		\param np Number of elements p.
		\param a Pointer to a.
		\param spa Step of p in a.
		\param b Pointer to b.
		\param spb Step of p in b.
	 **/
	static double x_p_p(
		size_t np,
		const double *a, size_t spa,
		const double *b, size_t spb);

	/**	\brief \f$ c_i = c_i + a_i b \f$
		\param ni Number of elements i.
		\param a Pointer to a.
		\param sia Step of i in a.
		\param b Scalar b.
		\param c Pointer to c.
		\param sic Step of i in c.
	 **/
	static void i_i_x(
		size_t ni,
		const double *a, size_t sia,
		double b,
		double *c, size_t sic);

	/**	\brief \f$ c_i = c_i + a_i b_i \f$
		\param ni Number of elements i.
		\param a Pointer to a.
		\param sia Step of i in a.
		\param b Pointer to b.
		\param sib Step of i in b.
		\param c Pointer to c.
		\param sic Step of i in c.
	 **/
	static void i_i_i(
		size_t ni,
		const double *a, size_t sia,
		const double *b, size_t sib,
		double *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL1_GENERIC_H
