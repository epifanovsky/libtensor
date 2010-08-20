#ifndef LIBTENSOR_LINALG_IMPL_GENERIC_H
#define LIBTENSOR_LINALG_IMPL_GENERIC_H

#include <cstdlib>
#include "../exception.h"

namespace libtensor {


/**	\brief Generic implementation of linear algebra

	This class provides a generic implementation for all linear algebra
	operations required for the functioning of libtensor. This
	implementation is guaranteed to work on any platform, which also makes
	it very inefficient. Platform-dependent implementations shall overload
	time-critical functions to improve the performance.

	\ingroup libtensor_linalg
 **/
class linalg_impl_generic {
public:
	static const char *k_clazz; //!< Class name

public:
	//!	\name Vector operations
	//@{

	/**	\brief Vector dot product: \f$ c = \sum_p a_p b_p \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param np Number of elements p.
		\param spa Step of p in a.
		\param spb Step of p in b.
		\return c.
	 **/
	static double x_p_p(const double *a, const double *b,
		size_t np, size_t spa, size_t spb);

	/**	\brief Vector add-multiply: \f$ c_i = c_i + a_i b \f$
		\param a Pointer to a.
		\param b Value of b.
		\param[in,out] c Pointer to c.
		\param ni Number of elements i.
		\param sia Step of i in a.
		\param sic Step of i in c.
	 **/
	static void i_i_x(const double *a, double b, double *c,
		size_t ni, size_t sia, size_t sic);

	//@}


	//!	\name Single matrix operations
	//@{

	/**	\brief Matrix transposition and scaling:
			\f$ c_{ij} = a_{ji} b \f$
	 **/
	static void ij_ji_x(const double *a, double b, double *c,
		size_t ni, size_t nj, size_t sic, size_t sja);

	//@}


	//!	\name Matrix-vector operations
	//@{

	/**	\brief Matrix-vector add-multiply:
			\f$ c_i = c_i + d \sum_p a_{ip} b_p \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param np Number of elements p.
		\param sia Step of i in a (sia >= np).
		\param sic Step of i in c (sic >= 1).
		\param spb Step of p in b (spb >= 1).
	 **/
	static void i_ip_p(const double *a, const double *b, double *c,
		double d, size_t ni, size_t np, size_t sia, size_t sic,
		size_t spb);

	/**	\brief Matrix-vector add-multiply:
			\f$ c_i = c_i + d \sum_p a_{pi} b_p \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param np Number of elements p.
		\param sic Step of i in c (sic >= 1).
		\param spa Step of p in a (spa >= ni).
		\param spb Step of p in b (spb >= 1).
	 **/
	static void i_pi_p(const double *a, const double *b, double *c,
		double d, size_t ni, size_t np, size_t sic, size_t spa,
		size_t spb);

	/**	\brief Matrix-vector direct product accumulation:
			\f$ c_{ij} = c_{ij} + d a_i b_j \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param Value of d.
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param sia Step of i in a (sia >= 1).
		\param sic Step of i in c (sic >= nj).
		\param sjb Step of j in b (sjb >= 1).
	 **/
	static void ij_i_j(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t sia, size_t sic,
		size_t sjb);

	//@}


	//!	\name Matrix-matrix operations
	//@{

	/**	\brief Matrix-matrix add-multiply:
			\f$ c_{ij} = c_{ij} + d \sum_p a_{ip} b_{jp} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param sia Step of i in a (sia >= np).
		\param sic Step of i in c (sic >= nj).
		\param sjb Step of j in b (sjb >= np).
	 **/
	static void ij_ip_jp(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sia,
		size_t sic, size_t sjb);

	/**	\brief Matrix-matrix add-multiply:
			\f$ c_{ij} = c_{ij} + d \sum_p a_{ip} b_{pj} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param sia Step of i in a (sia >= np).
		\param sic Step of i in c (sic >= nj).
		\param spb Step of p in b (spb >= nj);
	 **/
	static void ij_ip_pj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sia,
		size_t sic, size_t spb);

	/**	\brief Matrix-matrix add-multiply:
			\f$ c_{ij} = c_{ij} + d \sum_p a_{pi} b_{jp} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param sic Step of i in c (sic >= nj).
		\param sjb Step of j in b (sjb >= np).
		\param spa Step of p in a (spa >= ni).
	 **/
	static void ij_pi_jp(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sic,
		size_t sjb, size_t spa);

	/**	\brief Matrix-matrix add-multiply:
			\f$ c_{ij} = c_{ij} + d \sum_p a_{pi} b_{pj} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param sic Step of i in c (sic >= nj).
		\param spa Step of p in a (spa >= ni).
		\param spb Step of p in b (spb >= nj);
	 **/
	static void ij_pi_pj(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t sic,
		size_t spa, size_t spb);

	/**	\brief Dot product of two vectorized matrices:
			\f$ c = \sum_{pq} a_{pq} b_{qp} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param np Number of elements p.
		\param nq Number of elements q.
		\param spa Step of p in a (spa >= nq).
		\param sqb Step of q in b (sqb >= np).
		\return c.
	 **/
	static double x_pq_qp(const double *a, const double *b,
		size_t np, size_t nq, size_t spa, size_t sqb);

	//@}


	//!	\name Tensor-tensor contractions
	//@{

	/**	\brief Contraction: \f$ c_i = d \sum_{pq} a_{ipq} b_{qp} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param np Number of elements p.
		\param nq Number of elements q.
		\param sia Step of i in a (sia >= np * spa).
		\param sic Step of i in c (sic >= ni).
		\param spa Step of p in a (spa >= nq).
		\param sqb Step of q in b (sqb >= np).
	 **/
	static void i_ipq_qp(const double *a, const double *b, double *c,
		double d, size_t ni, size_t np, size_t nq, size_t sia,
		size_t sic, size_t spa, size_t sqb);

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

	/**	\brief Contraction: \f$ c_{ij} = d \sum_{pq} a_{piq} b_{pjq} \f$
		\param a Pointer to a.
		\param b Pointer to b.
		\param[in,out] c Pointer to c.
		\param d Value of d.
		\param ni Number of elements i.
		\param nj Number of elements j.
		\param np Number of elements p.
		\param nq Number of elements q.
		\param sia Step of i in a (sib >= nq).
		\param sic Step of i in c (sic >= nj).
		\param sjb Step of j in b (sjb >= nq).
		\param spa Step of p in a (spa >= nj * sja).
		\param spb Step of p in b (spb >= ni * sib).
	 **/
	static void ij_piq_pjq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t np, size_t nq,
		size_t sia, size_t sic, size_t sjb, size_t spa, size_t spb);

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
			\f$ c_{ijkl} = d \sum_{pq} a_{pilq} b_{pkjq} \f$
	 **/
	static void ijkl_pilq_pkjq(const double *a, const double *b, double *c,
		double d, size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
		size_t nq);

	//@}


protected:
	static void chkarg_ij_ipq_jqp(const double *a, const double *b,
		double *c, double d, size_t ni, size_t nj, size_t np, size_t nq,
		size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb)
		throw(bad_parameter);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IMPL_GENERIC_H
