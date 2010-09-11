#include "linalg_base_level1_generic.h"

namespace libtensor {


void linalg_base_level1_generic::i_x(
	size_t ni,
	double a,
	double *c, size_t sic) {

	for(size_t i = 0; i < ni; i++) c[i * sic] *= a;
}


double linalg_base_level1_generic::x_p_p(
	size_t np,
	const double *a, size_t spa,
	const double *b, size_t spb) {

	double c = 0.0;
	for(size_t p = 0; p < np; p++) c += a[p * spa] * b[p * spb];
	return c;
}


void linalg_base_level1_generic::i_i_x(
	size_t ni,
	const double *a, size_t sia,
	double b,
	double *c, size_t sic) {

	for(size_t i = 0; i < ni; i++) c[i * sic] += a[i * sia] * b;
}


void linalg_base_level1_generic::i_i_i(
	size_t ni,
	const double *a, size_t sia,
	const double *b, size_t sib,
	double *c, size_t sic) {

	for(size_t i = 0; i < ni; i++) c[i * sic] += a[i * sia] * b[i * sib];
}


} // namespace libtensor
