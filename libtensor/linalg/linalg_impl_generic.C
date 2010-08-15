#include "../exception.h"
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


void linalg_impl_generic::ij_ipq_jqp(const double *a, const double *b,
	double *c, double d, size_t ni, size_t nj, size_t np, size_t nq,
	size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb) {

	static const char *method = "ij_ipq_jqp()";

#ifdef LIBTENSOR_DEBUG
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

	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++) {
		c[i * sic + j] += d * x_pq_qp(a + i * sia, b + j * sjb, np, nq,
			spa, sqb);
	}
}


} // namespace libtensor
