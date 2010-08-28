#ifndef LIBTENSOR_LINALG_IMPL_MKL_H
#define LIBTENSOR_LINALG_IMPL_MKL_H

#include "linalg_impl_cblas.h"

namespace libtensor {


/**	\brief Implementation of linear algebra using Intel Math Kernel Library

	\sa linalg_impl_generic, linalg_impl_cblas

	\ingroup libtensor_linalg
 **/
class linalg_impl_mkl : public linalg_impl_cblas {
public:
	//!	\name Single matrix operations
	//@{

	static void ij_ij_x(const double *a, double b, double *c,
		size_t ni, size_t nj, size_t sia, size_t sic);

	static void ij_ji_x(const double *a, double b, double *c,
		size_t ni, size_t nj, size_t sic, size_t sja);

	//@}


	//!	\name Tensor-tensor contractions
	//@{

	/**	\brief Contraction: \f$ c_{ij} = d \sum_{pq} a_{ipq} b_{jqp} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param nq Number of elements q.
		\param sia Step of i in a (sia >= np * spa).
		\param sic Step of i in c (sic >= ni).
		\param sjb Step of j in b (sjb >= nq * sqb).
		\param spa Step of p in a (spa >= nq).
		\param sqb Step of q in b (sqb >= np).
	 **/
	static void ij_ipq_jqp(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t nq,
		size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb);

	//@}


	//!	\name Six-index tensor-tensor contractions
	//@{

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{iplq} b_{kpjq} \f$
	 **/
	static void ijkl_iplq_kpjq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

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

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{ipql} b_{pkqj} \f$
	 **/
	static void ijkl_ipql_pkqj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pilq} b_{kpjq} \f$
	 **/
	static void ijkl_pilq_kpjq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pilq} b_{pkjq} \f$
	 **/
	static void ijkl_pilq_pkjq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{piql} b_{kpqj} \f$
	 **/
	static void ijkl_piql_kpqj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{piql} b_{pkqj} \f$
	 **/
	static void ijkl_piql_pkqj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pkiq} b_{jplq} \f$
	 **/
	static void ijkl_pkiq_jplq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pkiq} b_{jpql} \f$
	 **/
	static void ijkl_pkiq_jpql(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pkiq} b_{pjlq} \f$
	 **/
	static void ijkl_pkiq_pjlq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pkiq} b_{pjql} \f$
	 **/
	static void ijkl_pkiq_pjql(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pliq} b_{jpkq} \f$
	 **/
	static void ijkl_pliq_jpkq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pliq} b_{jpqk} \f$
	 **/
	static void ijkl_pliq_jpqk(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	/**	\brief Contraction:
			\f$ c_{ijkl} = d \sum_{pq} a_{pliq} b_{pjqk} \f$
	 **/
	static void ijkl_pliq_pjqk(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	//@}


	//!	\name Memory operations
	//@{

	/**	\brief Allocates a temporary array of doubles
		\param n Array length.
		\return Pointer to the array.
	 **/
	static double *allocate(size_t n);

	/**	\brief Deallocates a temporary array previously allocated
			using allocate(size_t)
		\param p Pointer to the array.
	 **/
	static void deallocate(double *p);

	//@}


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IMPL_MKL_H
