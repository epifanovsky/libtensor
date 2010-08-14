#ifndef LIBTENSOR_LINALG_IMPL_CBLAS_H
#define LIBTENSOR_LINALG_IMPL_CBLAS_H

#include "linalg_impl_generic.h"

namespace libtensor {


/**	\brief Implementation of linear algebra using CBlas or GSL

	\sa linalg_impl_generic

	\ingroup libtensor_linalg
 **/
class linalg_impl_cblas : public linalg_impl_generic {
public:
	//!	\name Vector operations
	//@{

	static double x_p_p(const double *a, const double *b,
		size_t np, size_t spa, size_t spb);

	static void i_i_x(const double *a, double b, double *c,
		size_t ni, size_t sia, size_t sic);

	//@}


	//!	\name Matrix-vector operations
	//@{

	static void i_ip_p(const double *a, const double *b, double *c,
		double d, size_t ni, size_t np, size_t sia, size_t sic,
		size_t spb);

	static void i_pi_p(const double *a, const double *b, double *c,
		double d, size_t ni, size_t np, size_t sic, size_t spa,
		size_t spb);

	static void ij_i_j(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t sia, size_t sic,
		size_t sjb);

	//@}


	//!	\name Matrix-matrix operations
	//@{

	static void ij_ip_jp(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sia,
		size_t sic, size_t sjb);

	static void ij_ip_pj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sia,
		size_t sic, size_t spb);

	static void ij_pi_jp(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sic,
		size_t sjb, size_t spa);

	static void ij_pi_pj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sic,
		size_t spa, size_t spb);

	static double x_pq_qp(const double *a, const double *b,
		size_t np, size_t nq, size_t spa, size_t sqb);

	//@}

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IMPL_CBLAS_H
