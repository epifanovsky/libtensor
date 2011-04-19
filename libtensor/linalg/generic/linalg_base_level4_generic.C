#include "linalg_base_level2_generic.h"
#include "linalg_base_level4_generic.h"

namespace libtensor {


void linalg_base_level4_generic::ij_ipq_jqp_x(
	size_t ni, size_t nj, size_t np, size_t nq,
	const double *a, size_t spa, size_t sia,
	const double *b, size_t sqb, size_t sjb,
	double *c, size_t sic,
	double d) {

	for(size_t i = 0; i < ni; i++)
	for(size_t j = 0; j < nj; j++) {
		c[i * sic + j] += d * linalg_base_level2_generic::x_pq_qp(
			np, nq, a + i * sia, spa, b + j * sjb, sqb);
	}
}


} // namespace libtensor
