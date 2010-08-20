#include "linalg_impl_generic.h"

namespace libtensor {


const char *linalg_impl_generic::k_clazz = "linalg_impl_generic";


double linalg_impl_generic::x_p_p(const double *a, const double *b,
	size_t np, size_t spa, size_t spb) {

	double c = 0.0;
	for(size_t p = 0; p < np; p++) c += a[p * spa] * b[p * spb];
	return c;
}


void linalg_impl_generic::i_i_x(const double *a, double b, double *c,
	size_t ni, size_t sia, size_t sic) {

	for(size_t i = 0; i < ni; i++) c[i * sic] += a[i * sia] * b;
}


void linalg_impl_generic::ij_ji_x(const double *a, double b, double *c,
	size_t ni, size_t nj, size_t sic, size_t sja) {

	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++) {
		c[i * sic + j] = b * a[j * sja + i];
	}
}


void linalg_impl_generic::i_ip_p(const double *a, const double *b, double *c,
	double d, size_t ni, size_t np, size_t sia, size_t sic, size_t spb) {

	for(size_t i = 0; i < ni; i++) {
		double ci = 0.0;
		for(size_t p = 0; p < np; p++) {
			ci += a[i * sia + p] * b[p * spb];
		}
		c[i * sic] += d * ci;
	}
}


void linalg_impl_generic::i_pi_p(const double *a, const double *b, double *c,
	double d, size_t ni, size_t np, size_t sic, size_t spa, size_t spb) {

	for(size_t p = 0; p < np; p++)
	for(size_t i = 0; i < ni; i++) {
		c[i * sic] += d * a[p * spa + i] * b[p * spb];
	}
}


void linalg_impl_generic::ij_i_j(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t sia, size_t sic, size_t sjb) {

	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++) {
		c[i * sic + j] += d * a[i * sia] * b[j * sjb];
	}
}


void linalg_impl_generic::ij_ip_jp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sia, size_t sic,
	size_t sjb) {

	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++) {
		double cij = 0.0;
		for(size_t p = 0; p < np; p++) {
			cij += a[i * sia + p] * b[j * sjb + p];
		}
		c[i * sic + j] += d * cij;
	}
}


void linalg_impl_generic::ij_ip_pj(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sia, size_t sic,
	size_t spb) {

	for(size_t i = 0; i < ni; i++)
	for(size_t p = 0; p < np; p++) {
		double aip = a[i * sia + p];
		for(size_t j = 0; j < nj; j++) {
			c[i * sic + j] += d * aip * b[p * spb + j];
		}
	}
}


void linalg_impl_generic::ij_pi_jp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sic, size_t sjb,
	size_t spa) {

	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++)
	for(size_t p = 0; p < np; p++) {
		c[i * sic + j] += d * a[p * spa + i] * b[j * sjb + p];
	}
}


void linalg_impl_generic::ij_pi_pj(const double *a, const double *b, double *c,
	double d, size_t ni, size_t nj, size_t np, size_t sic, size_t spa,
	size_t spb) {

	for(size_t p = 0; p < np; p++)
	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++) {
		c[i * sic + j] += d * a[p * spa + i] * b[p * spb + j];
	}
}


double linalg_impl_generic::x_pq_qp(const double *a, const double *b,
	size_t np, size_t nq, size_t spa, size_t sqb) {

	double c = 0.0;
	for(size_t p = 0; p < np; p++)
	for(size_t q = 0; q < nq; q++) {
		c += a[p * spa + q] * b[q * sqb + p];
	}
	return c;
}


void linalg_impl_generic::i_ipq_qp(const double *a, const double *b, double *c,
	double d, size_t ni, size_t np, size_t nq, size_t sia, size_t sic,
	size_t spa, size_t sqb) {

	for(size_t i = 0; i < ni; i++) {
		c[i * sic] += d * x_pq_qp(a + i * sia, b, np, nq, spa, sqb);
	}
}


void linalg_impl_generic::chkarg_ij_ipq_jqp(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t np, size_t nq,
	size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb)
	throw(bad_parameter) {

	static const char *method = "chkarg_ij_ipq_jqp()";

#ifdef LIBTENSOR_DEBUG
	if(a == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"a");
	}
	if(b == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"b");
	}
	if(c == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"c");
	}
	if(ni == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"ni");
	}
	if(nj == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"nj");
	}
	if(np == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"np");
	}
	if(nq == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"nq");
	}
	if(sia < spa * np) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"sia");
	}
	if(sic < nj) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"sic");
	}
	if(sjb < sqb * nq) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"sjb");
	}
	if(spa < nq) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"spa");
	}
	if(sqb < np) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"sqb");
	}
#endif // LIBTENSOR_DEBUG
}


void linalg_impl_generic::ij_ipq_jqp(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t np, size_t nq,
	size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb) {

#ifdef LIBTENSOR_DEBUG
	chkarg_ij_ipq_jqp(a, b, c, d, ni, nj, np, nq, sia, sic, sjb, spa, sqb);
#endif // LIBTENSOR_DEBUG

	if(d == 0.0) return;

	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++) {
		c[i * sic + j] += d * x_pq_qp(a + i * sia, b + j * sjb, np, nq,
			spa, sqb);
	}
}


void linalg_impl_generic::ij_piq_pjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t np, size_t nq,
	size_t sia, size_t sic, size_t sjb, size_t spa, size_t spb) {

	for(size_t p = 0; p < np; p++) {
		ij_ip_jp(a + p * spa, b + p * spb, c, d, ni, nj, nq,
			sia, sic, sjb);
	}
}


void linalg_impl_generic::ijkl_iplq_kpjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nl * nq;
		const double *b1 = b + (k * np + p) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[lq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_impl_generic::ijkl_iplq_pkjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nl * nq;
		const double *b1 = b + (p * nk + k) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[lq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_impl_generic::ijkl_iplq_pkqj(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nl * nq;
		const double *b1 = b + (p * nk + k) * nq * nj;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t qj = q * nj + j;
				c[ijk + l] += d * a1[lq] * b1[qj];
			}
		}
		}
	}
	}
	}
}


void linalg_impl_generic::ijkl_pilq_pkjq(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t nk, size_t nl,
	size_t np, size_t nq) {

	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * ni + i) * nl * nq;
		const double *b1 = b + (p * nk + k) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[lq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


} // namespace libtensor
