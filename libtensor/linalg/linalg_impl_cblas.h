#ifndef LIBTENSOR_LINALG_IMPL_CBLAS_H
#define LIBTENSOR_LINALG_IMPL_CBLAS_H

#include "linalg_impl_generic.h"
#include "algo_ijkl_iplq_pkjq.h"
#include "algo_ijkl_iplq_pkqj.h"

namespace libtensor {


/**	\brief Implementation of linear algebra using CBlas or GSL

	\sa linalg_impl_generic

	\ingroup libtensor_linalg
 **/
class linalg_impl_cblas : public linalg_impl_generic {

	friend void algo_ijkl_iplq_pkjq<linalg_impl_cblas>(const double*,
		const double*, double*, double, size_t, size_t, size_t, size_t,
		size_t, size_t);
	friend void algo_ijkl_iplq_pkqj<linalg_impl_cblas>(const double*,
		const double*, double*, double, size_t, size_t, size_t, size_t,
		size_t, size_t);

public:
	//!	\name Vector operations
	//@{

	static double x_p_p(const double *a, const double *b,
		size_t np, size_t spa, size_t spb);

	static void i_i_x(const double *a, double b, double *c,
		size_t ni, size_t sia, size_t sic);

	//@}


	//!	\name Single matrix operations
	//@{

	static void ij_ij_x(const double *a, double b, double *c,
		size_t ni, size_t nj, size_t sia, size_t sic);

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


	//!	\name Six-index tensor-tensor contractions
	//@{

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{iplq} b_{pkjq} \f$
	 **/
	static void ijkl_iplq_pkjq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{iplq} b_{pkqj} \f$
	 **/
	static void ijkl_iplq_pkqj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	//@}


private:
	//!	\name Memory operations
	//@{

	/**	\brief Allocates a temporary array of doubles
		\param n Array length.
		\return Pointer to the array.
	 **/
	static double *allocate(size_t n) {
		return new double[n];
	}

	/**	\brief Deallocates a temporary array previously allocated
			using allocate(size_t)
		\param p Pointer to the array.
	 **/
	static void deallocate(double *p) {
		delete [] p;
	}

	//@}

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IMPL_CBLAS_H
