#if defined(USE_CBLAS) || defined(USE_MKL) || defined(USE_GSL)

#if defined(USE_CBLAS)
#include <cblas.h>
#elif defined(USE_MKL)
#include <mkl.h>
#elif defined(USE_GSL)
#include <gsl/gsl_cblas.h>
#endif

#include "linalg_impl_cblas.h"

namespace libtensor {


double linalg_impl_cblas::x_p_p(const double *a, const double *b,
	size_t np, size_t stpa, size_t stpb) {

	return cblas_ddot(np, a, stpa, b, stpb);
}


void linalg_impl_cblas::i_i_x(const double *a, double b, double *c,
	size_t ni, size_t sia, size_t sic) {

	cblas_daxpy(ni, b, a, sia, c, sic);
}


void linalg_impl_cblas::i_ip_p(const double *a, const double *b, double *c,
	double d, size_t ni, size_t np, size_t sia, size_t sic, size_t spb) {

	cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b, spb, 1.0,
		c, sic);
}


void linalg_impl_cblas::i_pi_p(const double *a, const double *b, double *c,
	double d, size_t ni, size_t np, size_t sic, size_t spa, size_t spb) {

	cblas_dgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b, spb, 1.0,
		c, sic);
}


void linalg_impl_cblas::ij_i_j(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t sia, size_t sic, size_t sjb) {

	cblas_dger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
}


void linalg_impl_cblas::ij_ip_jp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sia, size_t sic,
	size_t sjb) {

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ni, nj, np, d,
		a, sia, b, sjb, 1.0, c, sic);
}


void linalg_impl_cblas::ij_ip_pj(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sia, size_t sic,
	size_t spb) {

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, np, d,
		a, sia, b, spb, 1.0, c, sic);
}


void linalg_impl_cblas::ij_pi_jp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sic, size_t sjb,
	size_t spa) {

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, ni, nj, np, d,
		a, spa, b, sjb, 1.0, c, sic);
}


void linalg_impl_cblas::ij_pi_pj(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sic, size_t spa,
	size_t spb) {

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ni, nj, np, d,
		a, spa, b, spb, 1.0, c, sic);
}


double linalg_impl_cblas::x_pq_qp(const double *a, const double *b,
	size_t np, size_t nq, size_t spa, size_t sqb) {

	double c = 0.0;
	for(size_t q = 0; q < nq; q++) {
		c += cblas_ddot(np, a + q, spa, b + q * sqb, 1);
	}
	return c;
}


} // namespace libtensor

#endif // defined(USE_CBLAS) || defined(USE_MKL) || defined(USE_GSL)
