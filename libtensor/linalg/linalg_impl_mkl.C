#if defined(USE_MKL)

#include <mkl.h>
#ifdef HAVE_MKL_DOMATCOPY
#include <mkl_trans.h>
#endif // HAVE_MKL_DOMATCOPY
#include "linalg_impl_mkl.h"
#include "algo_ijkl_ipl_kpj.h"
#include "algo_ijkl_iplq_kpjq.h"
#include "algo_ijkl_iplq_pkjq.h"
#include "algo_ijkl_iplq_pkqj.h"
#include "algo_ijkl_ipql_pkqj.h"
#include "algo_ijkl_pilq_kpjq.h"
#include "algo_ijkl_pilq_pkjq.h"
#include "algo_ijkl_piql_kpqj.h"
#include "algo_ijkl_piql_pkqj.h"
#include "algo_ijkl_pkiq_jplq.h"
#include "algo_ijkl_pkiq_jpql.h"
#include "algo_ijkl_pkiq_pjlq.h"
#include "algo_ijkl_pkiq_pjql.h"
#include "algo_ijkl_pliq_jpkq.h"
#include "algo_ijkl_pliq_jpqk.h"
#include "algo_ijkl_pliq_pjqk.h"

namespace libtensor {


void linalg_impl_mkl::ij_ij_x(const double *a, double b, double *c, size_t ni,
	size_t nj, size_t sia, size_t sic) {
#ifdef HAVE_MKL_DOMATCOPY
	mkl_domatcopy('R', 'N', ni, nj, b, a, sia, c, sic);
#else
	linalg_impl_cblas::ij_ij_x(a, b, c, ni, nj, sia, sic);
#endif // HAVE_MKL_DOMATCOPY
}


void linalg_impl_mkl::ij_ji_x(const double *a, double b, double *c, size_t ni,
	size_t nj, size_t sic, size_t sja) {
#ifdef HAVE_MKL_DOMATCOPY
	mkl_domatcopy('R', 'T', nj, ni, b, a, sja, c, sic);
#else
	linalg_impl_cblas::ij_ji_x(a, b, c, ni, nj, sic, sja);
#endif // HAVE_MKL_DOMATCOPY
}


void linalg_impl_mkl::ij_ipq_jqp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t nq, size_t sia,
	size_t sic, size_t sjb, size_t spa, size_t sqb) {

	if(d == 0) return;

	size_t npq = np * nq;

	//	One operand needs to be transposed, the other may have to
	//	get its p and q concatenated.
	//	Use strategy:
	//	1. Avoid pq-concatenation.
	//	2. Minimize the number of transposed elements.
	//	3. Choose smallest-span argument.

	bool trpa = ni < nj; // Transpose a (true) or b (false)
	if(ni == nj) trpa = sia * ni < sjb * nj;
	if(trpa) {
		if(sjb != npq && sia == npq) trpa = false;
	} else {
		if(sia != npq && sjb == npq) trpa = true;
	}

	double *a1 = 0, *b1 = 0;
	const double *a2 = 0, *b2 = 0;
	size_t sia2, sjb2;

	//	Prepare arguments

	if(trpa) {
		//	Transpose a_ipq -> a_iqp
		//	Align pq nicely
		size_t sia1 = (npq % 4 == 0) ? npq : npq + (4 - npq % 4);
		a1 = allocate(ni * sia1);
		for(size_t i = 0; i < ni; i++) {
			ij_ji_x(a + i * sia, d, a1 + i * sia1,
				nq, np, np, spa);
		}
		a2 = a1;
		sia2 = sia1;

		//	Indexes qp in b_jqp may need to be concatenated
		size_t sjb1 = sia1;
		if(sqb != np) {
			b1 = allocate(nj * sjb1);
			for(size_t j = 0; j < nj; j++) {
				ij_ij_x(b + j * sjb, 1.0, b1 + j * sjb1,
					nq, np, sqb, np);
			}
			b2 = b1;
			sjb2 = sjb1;
		} else {
			b2 = b;
			sjb2 = sjb;
		}
	} else {
		//	Transpose b_jqp -> b_jpq
		//	Align pq nicely
		size_t sjb1 = (npq % 4 == 0) ? npq : npq + (4 - npq % 4);
		b1 = allocate(nj * sjb1);
		for(size_t j = 0; j < nj; j++) {
			ij_ji_x(b + j * sjb, d, b1 + j * sjb1,
				np, nq, nq, sqb);
		}
		b2 = b1;
		sjb2 = sjb1;

		//	Indexes pq in a_ipq may need to be concatenated
		size_t sia1 = sjb1;
		if(spa != nq) {
			a1 = allocate(ni * sia1);
			for(size_t i = 0; i < ni; i++) {
				ij_ij_x(a + i * sia, 1.0, a1 + i * sia1,
					np, nq, spa, nq);
			}
			a2 = a1;
			sia2 = sia1;
		} else {
			a2 = a;
			sia2 = sia;
		}
	}

	//	c_ij = a_iqp b_jqp
	ij_ip_jp(a2, b2, c, 1.0, ni, nj, npq, sia2, sic, sjb2);

	if(a1 != 0) deallocate(a1);
	if(b1 != 0) deallocate(b1);
}


void linalg_impl_cblas::ijkl_ipl_kpj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np) {

	algo_ijkl_ipl_kpj<linalg_impl_mkl>(a, b, c, d, ni, nj, nk, nl, np);
}


void linalg_impl_cblas::ijkl_iplq_kpjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_iplq_kpjq<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_iplq_pkjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_iplq_pkjq<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_iplq_pkqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_iplq_pkqj<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_ipql_pkqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_ipql_pkqj<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pilq_kpjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pilq_kpjq<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pilq_pkjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pilq_pkjq<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_piql_kpqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_piql_kpqj<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_piql_pkqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_piql_pkqj<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pkiq_jplq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_jplq<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pkiq_jpql(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_jpql<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pkiq_pjlq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_pjlq<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pkiq_pjql(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_pjql<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pliq_jpkq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pliq_jpkq<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pliq_jpqk(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pliq_jpqk<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_cblas::ijkl_pliq_pjqk(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pliq_pjqk<linalg_impl_mkl>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


double *linalg_impl_mkl::allocate(size_t n) {
#ifdef HAVE_MKL_MALLOC
	return (double*)MKL_malloc(n * sizeof(double), 4 * sizeof(double));
#else
	return new double[n];
#endif // HAVE_MKL_MALLOC
}


void linalg_impl_mkl::deallocate(double *p) {
#ifdef HAVE_MKL_MALLOC
	MKL_free(p);
#else
	delete [] p;
#endif // HAVE_MKL_MALLOC
}


} // namespace libtensor

#endif // defined(USE_MKL)

