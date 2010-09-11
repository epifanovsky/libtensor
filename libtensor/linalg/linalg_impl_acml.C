#if defined(USE_ACML)

#include <acml.h>

#include "linalg_impl_acml.h"
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


double linalg_impl_acml::x_p_p(const double *a, const double *b,
	size_t np, size_t stpa, size_t stpb) {

	return ddot(np, (double*)a, stpa, (double*)b, stpb);
}


void linalg_impl_acml::i_i_x(const double *a, double b, double *c,
	size_t ni, size_t sia, size_t sic) {

	daxpy(ni, b, (double*)a, sia, c, sic);
}


void linalg_impl_acml::ij_ij_x(const double *a, double b, double *c,
	size_t ni, size_t nj, size_t sia, size_t sic) {

	for(size_t i = 0; i < ni; i++) {
		dcopy(nj, (double*)a + i * sia, 1, c + i * sic, 1);
	}
}


void linalg_impl_acml::i_ip_p(const double *a, const double *b, double *c,
	double d, size_t ni, size_t np, size_t sia, size_t sic, size_t spb) {

	dgemv('T', np, ni, d, (double*)a, sia, (double*)b, spb, 1.0, c, sic);
}


void linalg_impl_acml::i_pi_p(const double *a, const double *b, double *c,
	double d, size_t ni, size_t np, size_t sic, size_t spa, size_t spb) {

	dgemv('N', ni, np, d, (double*)a, spa, (double*)b, spb, 1.0, c, sic);
}


void linalg_impl_acml::ij_i_j(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t sia, size_t sic, size_t sjb) {

	dger(nj, ni, d, (double*)b, sjb, (double*)a, sia, c, sic);
}


void linalg_impl_acml::ij_ip_jp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sia, size_t sic,
	size_t sjb) {

	dgemm('T', 'N', nj, ni, np, d, (double*)b, sjb, (double*)a, sia, 1.0, c, sic);
//	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ni, nj, np, d,
//		a, sia, b, sjb, 1.0, c, sic);
}


void linalg_impl_acml::ij_ip_pj(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sia, size_t sic,
	size_t spb) {

	dgemm('N', 'N', nj, ni, np, d, (double*)b, spb, (double*)a, sia, 1.0, c, sic);
//	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, np, d,
//		a, sia, b, spb, 1.0, c, sic);
}


void linalg_impl_acml::ij_pi_jp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sic, size_t sjb,
	size_t spa) {

	dgemm('T', 'T', nj, ni, np, d, (double*)b, sjb, (double*)a, spa, 1.0, c, sic);
//	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, ni, nj, np, d,
//		a, spa, b, sjb, 1.0, c, sic);
}


void linalg_impl_acml::ij_pi_pj(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sic, size_t spa,
	size_t spb) {

	dgemm('N', 'T', nj, ni, np, d, (double*)b, spb, (double*)a, spa, 1.0, c, sic);
//	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ni, nj, np, d,
//		a, spa, b, spb, 1.0, c, sic);
}


double linalg_impl_acml::x_pq_qp(const double *a, const double *b,
	size_t np, size_t nq, size_t spa, size_t sqb) {

	double c = 0.0;
	for(size_t q = 0; q < nq; q++) {
		c += ddot(np, (double*)a + q, spa, (double*)b + q * sqb, 1);
	}
	return c;
}


void linalg_impl_acml::ijkl_ipl_kpj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np) {

	algo_ijkl_ipl_kpj<linalg_impl_acml>(a, b, c, d, ni, nj, nk, nl, np);
}


void linalg_impl_acml::ijkl_iplq_kpjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_iplq_kpjq<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_iplq_pkjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_iplq_pkjq<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_iplq_pkqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_iplq_pkqj<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_ipql_pkqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_ipql_pkqj<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pilq_kpjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pilq_kpjq<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pilq_pkjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pilq_pkjq<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_piql_kpqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_piql_kpqj<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_piql_pkqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_piql_pkqj<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pkiq_jplq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_jplq<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pkiq_jpql(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_jpql<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pkiq_pjlq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_pjlq<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pkiq_pjql(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pkiq_pjql<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pliq_jpkq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pliq_jpkq<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pliq_jpqk(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pliq_jpqk<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


void linalg_impl_acml::ijkl_pliq_pjqk(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	algo_ijkl_pliq_pjqk<linalg_impl_acml>(a, b, c, d,
		ni, nj, nk, nl, np, nq);
}


} // namespace libtensor

#endif // defined(USE_ACML)
